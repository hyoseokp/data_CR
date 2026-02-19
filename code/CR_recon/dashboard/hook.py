"""
Dashboard Hook: Trainer callback
매 epoch마다 학습 데이터를 수집하여 대시보드 서버로 전송한다.
"""
import numpy as np
import torch

# 대시보드로 전송할 최대 샘플 수
MAX_DASHBOARD_SAMPLES = 8


class DashboardHook:
    """
    Trainer의 callback으로 등록되는 클래스.
    매 epoch마다 val/train sample 여러 개를 수집하여 대시보드로 전송.
    """

    def __init__(self, trainer):
        """
        trainer: Trainer 인스턴스 (dashboard_server, val_loader, device 접근)
        """
        self.trainer = trainer

    def __call__(self, epoch, train_loss, val_loss, best_val_loss, model, optimizer, trainer):
        """
        Trainer callback signature.
        매 epoch마다 val/train 샘플 여러 개를 수집하여 대시보드로 전송.
        """
        if not hasattr(trainer, 'dashboard_server') or trainer.dashboard_server is None:
            return

        if not trainer.dashboard_server.is_running():
            return

        try:
            # Wavelength 배열 계산 (공통)
            waves301 = np.linspace(400, 700, 301, dtype=np.float32)
            out_len = trainer.cfg["data"]["out_len"]
            edges = np.linspace(0, 301, out_len + 1).round().astype(int)
            waves = []
            for i in range(out_len):
                a, b = edges[i], edges[i + 1]
                if b <= a:
                    b = min(a + 1, 301)
                waves.append(float(waves301[a:b].mean()))

            # Val 샘플 추출 (여러 개)
            val_samples = self._extract_samples(trainer, trainer.val_loader, model, MAX_DASHBOARD_SAMPLES)

            # Train 샘플 추출 (여러 개)
            train_samples = self._extract_samples(trainer, trainer.train_loader, model, MAX_DASHBOARD_SAMPLES)

            if not val_samples or not train_samples:
                return

            # Loss component 통계 수집 (dashboard breakdown용)
            loss_stats = {}
            if hasattr(trainer, 'last_loss_stats'):
                loss_stats["train"] = {k: float(v) for k, v in trainer.last_loss_stats.items()}
            if hasattr(trainer, 'last_val_loss_stats'):
                loss_stats["val"] = {k: float(v) for k, v in trainer.last_val_loss_stats.items()}

            # 현재 실제 loss weight 값 (런타임 변경 반영)
            live_weights = {}
            if hasattr(trainer, 'loss_fn'):
                for attr in ('w_mse', 'w_mae', 'w_rel', 'w_grad', 'w_grad_mae', 'w_curv_mae'):
                    if hasattr(trainer.loss_fn, attr):
                        live_weights[attr] = float(getattr(trainer.loss_fn, attr))

            # 데이터 구성 (하위 호환: val/train은 첫 번째 샘플)
            data = {
                "epoch": int(epoch),
                "total_epochs": int(trainer.cfg["training"]["epochs"]),
                "lr": float(optimizer.param_groups[0]["lr"]),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "best_val": float(best_val_loss),
                "train_losses": [float(x) for x in trainer.train_losses],
                "val_losses": [float(x) for x in trainer.val_losses],
                "model_name": trainer.cfg["model"]["name"],
                "model_params": trainer.cfg["model"]["params"],
                "loss_name": trainer.cfg["loss"]["name"],
                "loss_params": trainer.cfg["loss"]["params"],
                "loss_stats": loss_stats,
                "live_weights": live_weights,
                "data_stats": {
                    "train_size": trainer.data_stats["train_size"],
                    "val_size": trainer.data_stats["val_size"],
                    "total_size": trainer.data_stats["total_size"],
                    "batch_size": trainer.data_stats["batch_size"],
                },
                "samples": {
                    "val": val_samples[0],          # 하위 호환
                    "train": train_samples[0],      # 하위 호환
                    "val_list": val_samples,         # 새 형식: 여러 샘플
                    "train_list": train_samples,     # 새 형식: 여러 샘플
                    "waves": waves
                }
            }

            # Dashboard로 전송
            trainer.dashboard_server.push_update(data)

        except Exception as e:
            trainer.log(f"[DASHBOARD HOOK ERROR] {e}", also_console=False)

    def _extract_samples(self, trainer, dataloader, model, num_samples=8):
        """
        데이터로더에서 첫 번째 배치의 여러 샘플을 추출하고 prediction + error 계산.

        Args:
            num_samples: 추출할 최대 샘플 수

        Returns:
            list of dict, 각 dict에 input_struct, gt_bggr, pred_bggr, abs_error_bggr
        """
        try:
            with torch.no_grad():
                for batch_struct, batch_gt in dataloader:
                    batch_struct = batch_struct.to(trainer.device, non_blocking=True)
                    batch_gt = batch_gt.to(trainer.device, non_blocking=True)

                    n = min(num_samples, batch_struct.size(0))

                    # Model inference (한 번에 n개)
                    pred_batch = model(batch_struct[:n])

                    results = []
                    for i in range(n):
                        s_struct = batch_struct[i].cpu().numpy()
                        s_gt = batch_gt[i].cpu().numpy()
                        s_pred = pred_batch[i].cpu().numpy()

                        input_struct = s_struct[0].tolist()

                        gt_bggr = [[s_gt[0, 0, :].tolist(), s_gt[0, 1, :].tolist()],
                                   [s_gt[1, 0, :].tolist(), s_gt[1, 1, :].tolist()]]

                        pred_bggr = [[s_pred[0, 0, :].tolist(), s_pred[0, 1, :].tolist()],
                                     [s_pred[1, 0, :].tolist(), s_pred[1, 1, :].tolist()]]

                        abs_error = np.abs(s_pred - s_gt)
                        abs_error_bggr = [[abs_error[0, 0, :].tolist(), abs_error[0, 1, :].tolist()],
                                          [abs_error[1, 0, :].tolist(), abs_error[1, 1, :].tolist()]]

                        results.append({
                            "input_struct": input_struct,
                            "gt_bggr": gt_bggr,
                            "pred_bggr": pred_bggr,
                            "abs_error_bggr": abs_error_bggr
                        })

                    return results if results else None

            return None

        except Exception as e:
            trainer.log(f"[SAMPLE EXTRACTION ERROR] {e}", also_console=False)
            return None
