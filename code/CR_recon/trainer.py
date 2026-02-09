"""
Trainer: 학습 엔진
config 기반으로 모델/loss/데이터를 조립하여 학습을 실행한다.
"""
import os
import math
import re
import sys
import time
from pathlib import Path
from typing import Callable, List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from data import create_dataloaders
from models import get_model
from losses import get_loss
from dashboard import DashboardServer
from dashboard.hook import DashboardHook
from constraints import enforce_intensity_sum_range


class Trainer:
    """학습 엔진: config 기반 모델/loss/데이터 조립 및 학습 실행."""

    def __init__(self, cfg: Dict[str, Any], device: Optional[torch.device] = None):
        """
        cfg: 전체 config dict (common_context_prompt.md의 Config Schema)
        device: torch.device (기본: cuda if available else cpu)
        """
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 출력 디렉토리
        self.output_dir = Path(cfg["output"]["dir"])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 로그 파일
        self.log_file = self.output_dir / cfg["output"]["log_file"]

        # seed 설정
        seed = cfg.get("seed", 42)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # 데이터 로더
        self.train_loader, self.val_loader = create_dataloaders(cfg)

        # 모델
        model_name = cfg["model"]["name"]
        model_params = cfg["model"]["params"]
        self.model = get_model(model_name, **model_params).to(self.device)

        # Loss 함수
        loss_name = cfg["loss"]["name"]
        loss_params = cfg["loss"]["params"]
        self.loss_fn = get_loss(loss_name, **loss_params)

        # Output constraint configuration (see constraints.py + data/dataset.py sign convention)
        constraints_cfg = cfg.get("constraints", {}) if isinstance(cfg, dict) else {}
        self.constraint_sum_min = float(constraints_cfg.get("sum_min", 0.45))
        self.constraint_sum_max = float(constraints_cfg.get("sum_max", 0.95))
        # Enforce constraint in the same domain as the model output/target by default.
        # If your "physical intensity" is represented as `-pred`, set physical_is_negative=True in config.
        self.constraint_physical_is_negative = bool(constraints_cfg.get("physical_is_negative", False))

        # Optimizer
        lr = cfg["training"]["lr"]
        weight_decay = cfg["training"]["weight_decay"]
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            eps=1e-8
        )

        # Scheduler: cosine annealing with linear warmup (epoch 단위)
        total_epochs = cfg["training"]["epochs"]
        warmup_epochs = max(1, int(cfg["training"]["warmup_ratio"] * total_epochs))
        min_lr_ratio = cfg["training"].get("min_lr_ratio", 0.01)  # base lr의 1%

        def lr_lambda(epoch: int):
            if epoch < warmup_epochs:
                return float(epoch + 1) / float(warmup_epochs)
            t = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        # AMP (Automatic Mixed Precision)
        self.use_amp = cfg["training"]["use_amp"] and self.device.type == "cuda"
        if self.use_amp:
            self.scaler = torch.amp.GradScaler("cuda")
        else:
            self.scaler = None

        # Gradient clipping
        self.grad_clip = cfg["training"]["grad_clip"]

        # Checkpoint 관련
        self.ckpt_best = self.output_dir / f"{model_name}_best.pt"
        self.ckpt_last = self.output_dir / f"{model_name}_last.pt"
        self.save_every = cfg["training"]["save_every"]
        self.best_val_loss = float("inf")

        # Callbacks (대시보드 hook 등)
        self.callbacks: List[Callable] = []

        # 데이터 통계 계산
        self.data_stats = {
            "train_size": len(self.train_loader.dataset),
            "val_size": len(self.val_loader.dataset),
            "total_size": len(self.train_loader.dataset) + len(self.val_loader.dataset),
            "batch_size": cfg["data"]["batch_size"],
        }

        # Dashboard 서버
        self.dashboard_server = None
        if cfg.get("dashboard", {}).get("enabled", False):
            try:
                port = cfg["dashboard"]["port"]
                self.dashboard_server = DashboardServer(port=port)
                # 초기 데이터 통계 설정
                self.dashboard_server.state["data_stats"] = self.data_stats
                self.add_callback(DashboardHook(self))
            except Exception as e:
                self.log(f"[WARNING] Failed to initialize dashboard server: {e}", also_console=True)
                self.dashboard_server = None

        # 학습 통계
        self.train_losses = []
        self.val_losses = []
        self.current_epoch = 0

    def add_callback(self, fn: Callable):
        """Callback 함수 등록 (epoch별 호출)."""
        self.callbacks.append(fn)

    def log(self, msg: str, also_console: bool = True):
        """로그 파일 및 콘솔에 기록."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()  # 즉시 디스크에 저장
            os.fsync(f.fileno())  # OS 버퍼도 flush
        if also_console:
            print(line)

    def train_one_epoch(self) -> float:
        """한 epoch 학습, 평균 train loss 반환."""
        self.model.train()
        total_loss = 0.0
        comp_sum = {"l_abs": 0.0, "l_dc": 0.0, "l_shape": 0.0}
        comp_count = 0
        shape_kind_seen = None
        extra_sum = {"d1_err_mean": 0.0, "d1_err_max": 0.0}
        extra_count = 0
        corr_min = {"dp_norm_min": float("inf"), "dt_norm_min": float("inf"), "denom_min": float("inf"), "r_min": float("inf")}
        corr_sum = {"r_mean": 0.0, "dp_norm_mean": 0.0, "dt_norm_mean": 0.0, "r_finite_frac": 0.0, "shape_valid_frac": 0.0}
        corr_count = 0
        clamp_sum = {"frac_clamped": 0.0, "frac_low": 0.0, "frac_high": 0.0, "total_min": 0.0, "total_max": 0.0, "total_mean": 0.0}
        clamp_count = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {self.current_epoch+1}",
            leave=True,
            dynamic_ncols=True,
            mininterval=0.3,
            file=sys.stdout,
            disable=False,
        )

        total_batches = len(self.train_loader)
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = self.model(x)
                    out, cstats = enforce_intensity_sum_range(
                        out,
                        sum_min=self.constraint_sum_min,
                        sum_max=self.constraint_sum_max,
                        physical_is_negative=self.constraint_physical_is_negative,
                        return_stats=True,
                    )
                    loss = self.loss_fn(out, y)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self.model(x)
                out, cstats = enforce_intensity_sum_range(
                    out,
                    sum_min=self.constraint_sum_min,
                    sum_max=self.constraint_sum_max,
                    physical_is_negative=self.constraint_physical_is_negative,
                    return_stats=True,
                )
                loss = self.loss_fn(out, y)
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()

            total_loss += float(loss.item())

            # Loss component diagnostics (per-batch, averaged at epoch end)
            if hasattr(self.loss_fn, "get_last_stats"):
                try:
                    st = self.loss_fn.get_last_stats() or {}
                except Exception:
                    st = {}
                if shape_kind_seen is None and isinstance(st.get("shape_kind"), str):
                    shape_kind_seen = st.get("shape_kind")
                if "l_abs" in st and "l_dc" in st and "l_shape" in st:
                    comp_sum["l_abs"] += float(st["l_abs"])
                    comp_sum["l_dc"] += float(st["l_dc"])
                    comp_sum["l_shape"] += float(st["l_shape"])
                    comp_count += 1
                # Derivative error stats (smoothl1/mse shape mode)
                if "d1_err_mean" in st and "d1_err_max" in st and st["d1_err_mean"] == st["d1_err_mean"]:
                    extra_sum["d1_err_mean"] += float(st["d1_err_mean"])
                    extra_sum["d1_err_max"] += float(st["d1_err_max"])
                    extra_count += 1
                # Correlation stats (pearson shape mode)
                corr_seen = False
                for k in corr_sum.keys():
                    if k in st and st[k] == st[k]:  # not NaN
                        corr_sum[k] += float(st[k])
                        corr_seen = True
                for k in corr_min.keys():
                    if k in st and st[k] == st[k]:
                        corr_min[k] = min(corr_min[k], float(st[k]))
                        corr_seen = True
                if corr_seen:
                    corr_count += 1

            # Constraint clamp diagnostics
            if isinstance(cstats, dict) and cstats:
                for k in clamp_sum.keys():
                    if k in cstats and cstats[k] == cstats[k]:
                        clamp_sum[k] += float(cstats[k])
                clamp_count += 1

            lr_now = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "lr": f"{lr_now:.2e}"
            })

            # 배치 진행 상황 대시보드에 전송 (10 배치마다 1회, 마지막 배치는 항상 전송)
            should_send = (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches
            if should_send and self.dashboard_server and self.dashboard_server.is_running():
                self.dashboard_server.push_progress({
                    "stage": "train",
                    "epoch": self.current_epoch,
                    "total_epochs": self.cfg["training"]["epochs"],
                    "batch": batch_idx + 1,
                    "total_batches": total_batches,
                    "current_loss": float(loss.item()),
                    "lr": lr_now
                })

        avg_loss = total_loss / max(1, len(self.train_loader))
        if comp_count > 0 or clamp_count > 0:
            comp = {k: (comp_sum[k] / max(1, comp_count)) for k in comp_sum.keys()}
            corr = {k: (corr_sum[k] / max(1, corr_count)) for k in corr_sum.keys()}
            clamp = {k: (clamp_sum[k] / max(1, clamp_count)) for k in clamp_sum.keys()}
            extra = {k: (extra_sum[k] / max(1, extra_count)) for k in extra_sum.keys()}
            extra_bits = []
            if shape_kind_seen:
                extra_bits.append(f"shape_kind={shape_kind_seen}")
            if extra_count > 0:
                extra_bits.append(f"d1_mean={extra['d1_err_mean']:.3e} d1_max={extra['d1_err_max']:.3e}")
            if corr_count > 0:
                extra_bits.append(
                    f"r_mean={corr['r_mean']:.3e} r_min={corr_min['r_min']:.3e} "
                    f"r_finite={corr['r_finite_frac']:.3f} shape_valid={corr['shape_valid_frac']:.3f} "
                    f"dp_norm_min={corr_min['dp_norm_min']:.3e} dt_norm_min={corr_min['dt_norm_min']:.3e} denom_min={corr_min['denom_min']:.3e}"
                )
            msg = (
                "[LOSS_DIAG][train] "
                f"abs={comp['l_abs']:.3e} dc={comp['l_dc']:.3e} shape={comp['l_shape']:.3e} "
                + ((" ".join(extra_bits) + " ") if extra_bits else "")
                + f"clamp_frac={clamp['frac_clamped']:.3f} low={clamp['frac_low']:.3f} high={clamp['frac_high']:.3f} "
                + f"sum_mean={clamp['total_mean']:.3e} sum_min={clamp['total_min']:.3e} sum_max={clamp['total_max']:.3e}"
            )
            self.log(msg, True)
        return avg_loss

    @torch.no_grad()
    def validate(self) -> float:
        """검증, 평균 val loss 반환."""
        self.model.eval()
        total_loss = 0.0
        comp_sum = {"l_abs": 0.0, "l_dc": 0.0, "l_shape": 0.0}
        comp_count = 0
        shape_kind_seen = None
        extra_sum = {"d1_err_mean": 0.0, "d1_err_max": 0.0}
        extra_count = 0
        corr_min = {"dp_norm_min": float("inf"), "dt_norm_min": float("inf"), "denom_min": float("inf"), "r_min": float("inf")}
        corr_sum = {"r_mean": 0.0, "dp_norm_mean": 0.0, "dt_norm_mean": 0.0, "r_finite_frac": 0.0, "shape_valid_frac": 0.0}
        corr_count = 0
        clamp_sum = {"frac_clamped": 0.0, "frac_low": 0.0, "frac_high": 0.0, "total_min": 0.0, "total_max": 0.0, "total_mean": 0.0}
        clamp_count = 0

        pbar = tqdm(
            self.val_loader,
            desc="Validation",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.3,
            file=sys.stdout,
            disable=False,
        )

        total_batches = len(self.val_loader)
        for batch_idx, (x, y) in enumerate(pbar):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            out = self.model(x)
            out, cstats = enforce_intensity_sum_range(
                out,
                sum_min=self.constraint_sum_min,
                sum_max=self.constraint_sum_max,
                physical_is_negative=self.constraint_physical_is_negative,
                return_stats=True,
            )
            loss = self.loss_fn(out, y)
            total_loss += float(loss.item())

            if hasattr(self.loss_fn, "get_last_stats"):
                try:
                    st = self.loss_fn.get_last_stats() or {}
                except Exception:
                    st = {}
                if shape_kind_seen is None and isinstance(st.get("shape_kind"), str):
                    shape_kind_seen = st.get("shape_kind")
                if "l_abs" in st and "l_dc" in st and "l_shape" in st:
                    comp_sum["l_abs"] += float(st["l_abs"])
                    comp_sum["l_dc"] += float(st["l_dc"])
                    comp_sum["l_shape"] += float(st["l_shape"])
                    comp_count += 1
                if "d1_err_mean" in st and "d1_err_max" in st and st["d1_err_mean"] == st["d1_err_mean"]:
                    extra_sum["d1_err_mean"] += float(st["d1_err_mean"])
                    extra_sum["d1_err_max"] += float(st["d1_err_max"])
                    extra_count += 1
                corr_seen = False
                for k in corr_sum.keys():
                    if k in st and st[k] == st[k]:
                        corr_sum[k] += float(st[k])
                        corr_seen = True
                for k in corr_min.keys():
                    if k in st and st[k] == st[k]:
                        corr_min[k] = min(corr_min[k], float(st[k]))
                        corr_seen = True
                if corr_seen:
                    corr_count += 1

            if isinstance(cstats, dict) and cstats:
                for k in clamp_sum.keys():
                    if k in cstats and cstats[k] == cstats[k]:
                        clamp_sum[k] += float(cstats[k])
                clamp_count += 1

            # 배치 진행 상황 대시보드에 전송 (5 배치마다 1회, 마지막 배치는 항상 전송)
            should_send = (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches
            if should_send and self.dashboard_server and self.dashboard_server.is_running():
                self.dashboard_server.push_progress({
                    "stage": "val",
                    "epoch": self.current_epoch,
                    "total_epochs": self.cfg["training"]["epochs"],
                    "batch": batch_idx + 1,
                    "total_batches": total_batches,
                    "current_loss": float(loss.item())
                })

        avg_loss = total_loss / max(1, len(self.val_loader))
        if comp_count > 0 or clamp_count > 0:
            comp = {k: (comp_sum[k] / max(1, comp_count)) for k in comp_sum.keys()}
            corr = {k: (corr_sum[k] / max(1, corr_count)) for k in corr_sum.keys()}
            clamp = {k: (clamp_sum[k] / max(1, clamp_count)) for k in clamp_sum.keys()}
            extra = {k: (extra_sum[k] / max(1, extra_count)) for k in extra_sum.keys()}
            extra_bits = []
            if shape_kind_seen:
                extra_bits.append(f"shape_kind={shape_kind_seen}")
            if extra_count > 0:
                extra_bits.append(f"d1_mean={extra['d1_err_mean']:.3e} d1_max={extra['d1_err_max']:.3e}")
            if corr_count > 0:
                extra_bits.append(
                    f"r_mean={corr['r_mean']:.3e} r_min={corr_min['r_min']:.3e} "
                    f"r_finite={corr['r_finite_frac']:.3f} shape_valid={corr['shape_valid_frac']:.3f} "
                    f"dp_norm_min={corr_min['dp_norm_min']:.3e} dt_norm_min={corr_min['dt_norm_min']:.3e} denom_min={corr_min['denom_min']:.3e}"
                )
            msg = (
                "[LOSS_DIAG][val] "
                f"abs={comp['l_abs']:.3e} dc={comp['l_dc']:.3e} shape={comp['l_shape']:.3e} "
                + ((" ".join(extra_bits) + " ") if extra_bits else "")
                + f"clamp_frac={clamp['frac_clamped']:.3f} low={clamp['frac_low']:.3f} high={clamp['frac_high']:.3f} "
                + f"sum_mean={clamp['total_mean']:.3e} sum_min={clamp['total_min']:.3e} sum_max={clamp['total_max']:.3e}"
            )
            self.log(msg, True)
        return avg_loss

    def save_checkpoint(self, is_best: bool = False, periodic: bool = False):
        """체크포인트 저장."""
        ckpt = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.current_epoch,
            "val_loss": self.val_losses[-1] if self.val_losses else float("inf"),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }

        # Always write "last" so `--resume outputs/<model>_last.pt` truly resumes the latest state,
        # even when the current epoch also becomes the best checkpoint.
        torch.save(ckpt, self.ckpt_last)

        if is_best:
            torch.save(ckpt, self.ckpt_best)
            self.log(f"[CKPT] saved best to {self.ckpt_best}", also_console=True)

        if periodic:
            ckpt_periodic = self.output_dir / f"{self.cfg['model']['name']}_epoch_{self.current_epoch:04d}.pt"
            torch.save(ckpt, ckpt_periodic)
            self.log(f"[CKPT] saved periodic to {ckpt_periodic}", also_console=True)

    def load_checkpoint(self, ckpt_path: str):
        """체크포인트 로드."""
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.current_epoch = ckpt["epoch"]
        self.best_val_loss = ckpt.get("best_val_loss", float("inf"))
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        self.log(f"[CKPT] loaded from {ckpt_path} (epoch {self.current_epoch})", also_console=True)

    def load_weights(self, ckpt_path: str):
        """모델 가중치만 로드 (optimizer/epoch 초기화 안 함)."""
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.log(f"[WEIGHTS] loaded model weights from {ckpt_path}", also_console=True)

    def _parse_log_losses(self):
        """로그 파일에서 이전 train/val loss 히스토리를 복원."""
        train_losses, val_losses = [], []
        pattern = re.compile(r'train_loss=([\d.e+-]+)\s+val_loss=([\d.e+-]+)')
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    train_losses.append(float(m.group(1)))
                    val_losses.append(float(m.group(2)))
        return train_losses, val_losses

    def train(self, resume_from: Optional[str] = None, init_weights: Optional[str] = None):
        """전체 학습 루프."""
        is_resume = resume_from is not None

        # 모델 가중치만 로드 (epoch 0부터 새로 학습)
        if init_weights:
            self.load_weights(init_weights)

        # Resume: checkpoint 로드 (로그 파일보다 선행)
        if is_resume:
            self.load_checkpoint(resume_from)
            self.current_epoch += 1
            # Checkpoint에 loss 히스토리가 없으면 로그에서 복원
            if not self.train_losses and self.log_file.exists():
                self.train_losses, self.val_losses = self._parse_log_losses()
            # Scheduler를 현재 epoch까지 fast-forward
            for _ in range(self.current_epoch):
                self.scheduler.step()
        else:
            self.current_epoch = 0

        # 로그 파일: 새로 시작시 초기화, resume시 유지
        if not is_resume:
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write("")
                f.flush()
                os.fsync(f.fileno())

        # 상세 설정 정보 기록
        self.log("=" * 80, True)
        if is_resume:
            self.log(f"TRAINING RESUMED FROM EPOCH {self.current_epoch}", True)
        else:
            self.log("TRAINING CONFIGURATION", True)
        self.log("=" * 80, True)

        # Model 정보
        self.log(f"Model: {self.cfg['model']['name']}", True)
        self.log(f"Model Path: models/{self.cfg['model']['name']}.py", True)
        self.log(f"Model Parameters:", True)
        for param_name, param_value in self.cfg['model']['params'].items():
            self.log(f"  - {param_name}: {param_value}", False)

        # Loss 정보
        self.log(f"Loss Function: {self.cfg['loss']['name']}", True)
        self.log(f"Loss Path: losses/{self.cfg['loss']['name']}.py", True)
        self.log(f"Loss Parameters:", True)
        for param_name, param_value in self.cfg['loss']['params'].items():
            self.log(f"  - {param_name}: {param_value}", False)

        # Hyperparameters
        self.log(f"Hyperparameters:", True)
        self.log(f"  - Learning Rate: {self.cfg['training']['lr']}", False)
        self.log(f"  - Weight Decay: {self.cfg['training']['weight_decay']}", False)
        self.log(f"  - Epochs: {self.cfg['training']['epochs']}", False)
        self.log(f"  - Batch Size: {self.cfg['data']['batch_size']}", False)
        self.log(f"  - Gradient Clip: {self.cfg['training']['grad_clip']}", False)
        self.log(f"  - Warmup Ratio: {self.cfg['training']['warmup_ratio']}", False)
        self.log(f"  - Use AMP: {self.use_amp}", False)
        if self.device.type == "cuda":
            try:
                gpu_name = torch.cuda.get_device_name(0)
            except Exception:
                gpu_name = "unknown"
            self.log(f"  - Device: cuda ({gpu_name})", False)
        else:
            self.log(f"  - Device: {self.device}", False)
        self.log(
            f"  - Constraint Sum Range: [{self.constraint_sum_min}, {self.constraint_sum_max}] "
            f"(physical_is_negative={self.constraint_physical_is_negative})",
            False,
        )
        self.log(f"  - Augment 180: {self.cfg['data'].get('augment_180', False)}", False)

        # Data 정보
        self.log(f"Data Statistics:", True)
        self.log(f"  - Train Size: {self.data_stats['train_size']}", False)
        self.log(f"  - Val Size: {self.data_stats['val_size']}", False)
        self.log(f"  - Total Size: {self.data_stats['total_size']}", False)
        self.log("=" * 80, True)
        self.log("", True)

        # Dashboard 서버 시작
        if self.dashboard_server:
            try:
                self.dashboard_server.start()
                if not is_resume:
                    self.dashboard_server.reset_state()
                else:
                    # Resume 시 이전 loss 히스토리를 대시보드에 즉시 반영
                    self.dashboard_server.state["train_losses"] = [float(x) for x in self.train_losses]
                    self.dashboard_server.state["val_losses"] = [float(x) for x in self.val_losses]
                    self.dashboard_server.state["epoch"] = self.current_epoch - 1
                    self.dashboard_server.state["total_epochs"] = self.cfg["training"]["epochs"]
                    self.dashboard_server.state["best_val"] = float(self.best_val_loss)
                    if self.train_losses:
                        self.dashboard_server.state["train_loss"] = float(self.train_losses[-1])
                    if self.val_losses:
                        self.dashboard_server.state["val_loss"] = float(self.val_losses[-1])
                self.log("Dashboard server started", True)
            except Exception as e:
                self.log(f"[WARNING] Failed to start dashboard server: {e}", also_console=True)

        epochs = self.cfg["training"]["epochs"]

        lr_now = self.optimizer.param_groups[0]["lr"]
        self.log(f"Start training | epochs={epochs} lr={lr_now:.2e} "
                f"batch_size={self.cfg['data']['batch_size']} use_amp={self.use_amp}", True)
        self.log(f"Model: {self.cfg['model']['name']} | Loss: {self.cfg['loss']['name']}", True)

        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_one_epoch()
            self.train_losses.append(train_loss)

            # Validate
            val_loss = self.validate()
            self.val_losses.append(val_loss)

            # Update best
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint
            self.save_checkpoint(is_best=is_best, periodic=(epoch + 1) % self.save_every == 0)

            # Log
            lr_now = self.optimizer.param_groups[0]["lr"]
            self.log(f"[EPOCH] {epoch+1}/{epochs} train_loss={train_loss:.6e} val_loss={val_loss:.6e} "
                    f"best_val={self.best_val_loss:.6e} lr={lr_now:.2e}", True)

            # Scheduler step (epoch-based, not batch-based)
            self.scheduler.step()

            # Loss spike detection
            if train_loss > 1.0:
                self.log(f"[WARNING] High train loss detected: {train_loss:.6e}", also_console=True)
            if val_loss > 1.0:
                self.log(f"[WARNING] High val loss detected: {val_loss:.6e}", also_console=True)

            # Callbacks
            for callback in self.callbacks:
                try:
                    callback(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        best_val_loss=self.best_val_loss,
                        model=self.model,
                        optimizer=self.optimizer,
                        trainer=self
                    )
                except Exception as e:
                    self.log(f"[CALLBACK ERROR] {e}", also_console=True)

        self.log(f"Training complete! best_val_loss={self.best_val_loss:.6e}", True)

        # Dashboard 서버 종료
        if self.dashboard_server:
            try:
                self.dashboard_server.stop()
                self.log("Dashboard server stopped", True)
            except Exception as e:
                self.log(f"[WARNING] Failed to stop dashboard server: {e}", also_console=True)
