"""
Adaptive Spectral Loss v5 — MSE/MAE + Relative MSE + Derivative MSE/MAE + Curvature MAE.

v4 → v5 변경:
  - Curvature MAE 항 추가: w_curv_mae × |d²pred - d²tgt|.mean()
    → 2차 미분(곡률) 오차. 피크 꼭짓점의 shape을 직접 맞춤

  L = w_mse·L_mse + w_mae·L_mae + w_rel·L_rel + w_grad·L_grad_mse
      + w_grad_mae·L_grad_mae + w_curv_mae·L_curv_mae

Tensors follow project convention:
  pred, tgt: (B, 2, 2, L)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveSpectralLoss(nn.Module):
    def __init__(
        self,
        # 가중치
        w_mse: float = 1.0,
        w_mae: float = 0.0,            # v4 신규: MAE (L1)
        w_rel: float = 1.0,
        w_grad: float = 1.0,           # GradMSE
        w_grad_mae: float = 0.0,       # v4 신규: GradMAE (L1)
        w_curv_mae: float = 0.0,       # v5 신규: Curvature MAE (2차미분 L1)
        # Relative MSE 안정성 파라미터
        rel_eps: float = 1e-4,         # 분모 안정화 (0 나누기 방지)
        rel_max: float = 10.0,         # 상대오차 클리핑 (초기 발산 방지)
        # Blue channel 가중치
        blue_weight: float = 2.0,
        # Auto-balancing (EMA scale normalization) — 현재 비권장
        use_uncertainty: bool = False,
        # EMA momentum (0.99 = ~100 step 반감기)
        ema_momentum: float = 0.99,
        # --- 하위 호환용 (무시됨) ---
        w_sam: float = 1.0,
        w_ffl: float = 1.0,
        ffl_alpha: float = 1.0,
        ffl_use_full_spectrum: bool = True,
        sam_eps: float = 1e-7,
    ):
        super().__init__()
        self.w_mse = float(w_mse)
        self.w_mae = float(w_mae)
        self.w_rel = float(w_rel)
        self.w_grad = float(w_grad)
        self.w_grad_mae = float(w_grad_mae)
        self.w_curv_mae = float(w_curv_mae)
        self.rel_eps = float(rel_eps)
        self.rel_max = float(rel_max)
        self.blue_weight = float(blue_weight)
        self.auto_balance = use_uncertainty
        self.ema_momentum = float(ema_momentum)

        # EMA buffers
        if self.auto_balance:
            self.register_buffer("ema_mse", torch.tensor(-1.0))
            self.register_buffer("ema_rel", torch.tensor(-1.0))
            self.register_buffer("ema_grad", torch.tensor(-1.0))

        self._last_stats: dict = {}

    def get_last_stats(self) -> dict:
        return dict(self._last_stats) if isinstance(self._last_stats, dict) else {}

    def _blue_weight_tensor(self, ref: torch.Tensor) -> torch.Tensor:
        """BGGR 패턴에서 Blue 채널(1,1)에 추가 가중치."""
        w = torch.ones(1, 2, 2, 1, device=ref.device, dtype=ref.dtype)
        w[0, 1, 1, 0] = self.blue_weight
        return w

    def _weighted_mse(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Blue 가중치 적용된 MSE loss."""
        diff_sq = (pred - tgt) ** 2
        if self.blue_weight != 1.0:
            w = self._blue_weight_tensor(pred)
            diff_sq = diff_sq * w
        return diff_sq.mean()

    def _weighted_mae(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """Blue 가중치 적용된 MAE (L1) loss."""
        diff_abs = (pred - tgt).abs()
        if self.blue_weight != 1.0:
            w = self._blue_weight_tensor(pred)
            diff_abs = diff_abs * w
        return diff_abs.mean()

    def _relative_mse(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Relative MSE: ((pred - tgt) / (|tgt| + eps))²
        """
        rel_err = (pred - tgt) / (tgt.abs() + self.rel_eps)
        rel_err = rel_err.clamp(-self.rel_max, self.rel_max)
        rel_sq = rel_err ** 2
        if self.blue_weight != 1.0:
            w = self._blue_weight_tensor(pred)
            rel_sq = rel_sq * w
        return rel_sq.mean()

    def _derivative_mse(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Derivative MSE: 스펙트럼 1차 미분(인접 파장 간 차이)의 MSE.
        """
        dp = pred[..., 1:] - pred[..., :-1]  # (B, 2, 2, 300)
        dt = tgt[..., 1:] - tgt[..., :-1]
        diff_sq = (dp - dt) ** 2
        if self.blue_weight != 1.0:
            w = self._blue_weight_tensor(pred)
            diff_sq = diff_sq * w
        return diff_sq.mean()

    def _derivative_mae(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Derivative MAE: 스펙트럼 1차 미분(인접 파장 간 차이)의 MAE.
        constant gradient로 기울기 잔여 오차를 끝까지 추격.
        """
        dp = pred[..., 1:] - pred[..., :-1]  # (B, 2, 2, 300)
        dt = tgt[..., 1:] - tgt[..., :-1]
        diff_abs = (dp - dt).abs()
        if self.blue_weight != 1.0:
            w = self._blue_weight_tensor(pred)
            diff_abs = diff_abs * w
        return diff_abs.mean()

    def _curvature_mae(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
        Curvature MAE: 스펙트럼 2차 미분(곡률)의 MAE.
        d²f[i] = f[i+1] - 2·f[i] + f[i-1]
        피크 꼭짓점의 shape(뾰족함/둥근 정도)을 직접 맞춤.
        """
        d2p = pred[..., 2:] - 2 * pred[..., 1:-1] + pred[..., :-2]  # (B, 2, 2, L-2)
        d2t = tgt[..., 2:] - 2 * tgt[..., 1:-1] + tgt[..., :-2]
        diff_abs = (d2p - d2t).abs()
        if self.blue_weight != 1.0:
            w = self._blue_weight_tensor(pred)
            diff_abs = diff_abs * w
        return diff_abs.mean()

    def _update_ema(self, ema_buf: torch.Tensor, val: float) -> None:
        """EMA buffer 업데이트. 첫 호출 시 현재 값으로 초기화."""
        if ema_buf.item() < 0:
            ema_buf.fill_(val)
        else:
            ema_buf.mul_(self.ema_momentum).add_((1.0 - self.ema_momentum) * val)

    def forward(self, pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        pred = pred.float()
        tgt = tgt.float()
        pred = torch.clamp(pred, -1e3, 1e3)
        tgt = torch.clamp(tgt, -1e3, 1e3)

        # 1. MSE loss (절대값 정확도) — w_mse=0이면 스킵
        l_mse = self._weighted_mse(pred, tgt) if self.w_mse > 0 else torch.tensor(0.0, device=pred.device)

        # 2. MAE loss (constant gradient) — w_mae=0이면 스킵
        l_mae = self._weighted_mae(pred, tgt) if self.w_mae > 0 else torch.tensor(0.0, device=pred.device)

        # 3. Relative MSE (상대 오차 정확도)
        l_rel = self._relative_mse(pred, tgt) if self.w_rel > 0 else torch.tensor(0.0, device=pred.device)

        # 4. Derivative MSE (기울기 MSE) — w_grad=0이면 스킵
        l_grad = self._derivative_mse(pred, tgt) if self.w_grad > 0 else torch.tensor(0.0, device=pred.device)

        # 5. Derivative MAE (기울기 MAE) — w_grad_mae=0이면 스킵
        l_grad_mae = self._derivative_mae(pred, tgt) if self.w_grad_mae > 0 else torch.tensor(0.0, device=pred.device)

        # 6. Curvature MAE (2차미분 MAE) — w_curv_mae=0이면 스킵
        l_curv_mae = self._curvature_mae(pred, tgt) if self.w_curv_mae > 0 else torch.tensor(0.0, device=pred.device)

        if self.auto_balance:
            with torch.no_grad():
                self._update_ema(self.ema_mse, l_mse.item())
                self._update_ema(self.ema_rel, l_rel.item())
                self._update_ema(self.ema_grad, l_grad.item())

            eps = 1e-8
            l_mse_norm = l_mse / (self.ema_mse.detach() + eps)
            l_rel_norm = l_rel / (self.ema_rel.detach() + eps)
            l_grad_norm = l_grad / (self.ema_grad.detach() + eps)

            loss = (self.w_mse * l_mse_norm
                    + self.w_mae * l_mae
                    + self.w_rel * l_rel_norm
                    + self.w_grad * l_grad_norm
                    + self.w_grad_mae * l_grad_mae
                    + self.w_curv_mae * l_curv_mae)
        else:
            loss = (self.w_mse * l_mse
                    + self.w_mae * l_mae
                    + self.w_rel * l_rel
                    + self.w_grad * l_grad
                    + self.w_grad_mae * l_grad_mae
                    + self.w_curv_mae * l_curv_mae)

        # Stats for dashboard logging
        # trainer.py 호환: l_abs, l_dc, l_shape 키 유지
        #   l_abs  = MSE (절대 오차) — MAE만 쓰면 0이 되므로 l_mae도 추가
        #   l_dc   = Relative MSE (상대 오차)
        #   l_shape = Derivative MSE/MAE (기울기 오차)
        with torch.no_grad():
            stats = {
                "shape_kind": "adaptive",
                "l_abs": float(l_mse.detach().item()),
                "l_mae": float(l_mae.detach().item()),
                "l_dc": float(l_rel.detach().item()),
                "l_shape": float(l_grad.detach().item()),
                "l_grad_mae": float(l_grad_mae.detach().item()),
                "l_curv_mae": float(l_curv_mae.detach().item()),
            }
            if self.auto_balance:
                stats["uw_mse"] = float(self.ema_mse.item())
                stats["uw_rel"] = float(self.ema_rel.item())
                stats["uw_grad"] = float(self.ema_grad.item())
            self._last_stats = stats

        return torch.clamp(loss, 0, 100)


def get_adaptive_spectral_loss(
    w_mse: float = 1.0,
    w_mae: float = 0.0,
    w_rel: float = 1.0,
    w_grad: float = 1.0,
    w_grad_mae: float = 0.0,
    w_curv_mae: float = 0.0,
    rel_eps: float = 1e-4,
    rel_max: float = 10.0,
    blue_weight: float = 2.0,
    use_uncertainty: bool = False,
    ema_momentum: float = 0.99,
    # 하위 호환용 (무시됨, 기존 config YAML에서 에러 방지)
    w_sam: float = 1.0,
    w_ffl: float = 1.0,
    ffl_alpha: float = 1.0,
    ffl_use_full_spectrum: bool = True,
    sam_eps: float = 1e-7,
) -> AdaptiveSpectralLoss:
    """Factory function."""
    return AdaptiveSpectralLoss(
        w_mse=w_mse,
        w_mae=w_mae,
        w_rel=w_rel,
        w_grad=w_grad,
        w_grad_mae=w_grad_mae,
        w_curv_mae=w_curv_mae,
        rel_eps=rel_eps,
        rel_max=rel_max,
        blue_weight=blue_weight,
        use_uncertainty=use_uncertainty,
        ema_momentum=ema_momentum,
    )
