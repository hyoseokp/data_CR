"""
Dashboard Hook: Trainer callback
매 epoch마다 학습 데이터를 수집하여 대시보드 서버로 전송한다.
"""
import numpy as np
import torch


class DashboardHook:
    """
    Trainer의 callback으로 등록되는 클래스.
    매 epoch마다 val sample 1개를 수집하여 대시보드로 전송.
    """

    def __init__(self, trainer):
        """
        trainer: Trainer 인스턴스 (dashboard_server, val_loader, device 접근)
        """
        self.trainer = trainer

    def __call__(self, epoch, train_loss, val_loss, best_val_loss, model, optimizer, trainer):
        """
        Trainer callback signature.
        매 epoch마다 호출된다.
        """
        if not hasattr(trainer, 'dashboard_server') or trainer.dashboard_server is None:
            return

        if not trainer.dashboard_server.is_running():
            return

        try:
            # Val dataloader에서 첫 번째 배치 샘플 1개 추출
            sample_struct = None
            sample_gt = None
            sample_pred = None

            with torch.no_grad():
                for batch_struct, batch_gt in trainer.val_loader:
                    batch_struct = batch_struct.to(trainer.device, non_blocking=True)
                    batch_gt = batch_gt.to(trainer.device, non_blocking=True)

                    # 첫 번째 샘플만 사용
                    sample_struct = batch_struct[0].cpu().numpy()
                    sample_gt = batch_gt[0].cpu().numpy()

                    # Model inference
                    sample_struct_batch = batch_struct[:1]
                    sample_pred_batch = model(sample_struct_batch)
                    sample_pred = sample_pred_batch[0].cpu().numpy()

                    break  # 첫 배치 하나만

            if sample_struct is None or sample_gt is None or sample_pred is None:
                return

            # Wavelength 배열 계산
            waves301 = np.linspace(400, 700, 301, dtype=np.float32)
            out_len = trainer.cfg["data"]["out_len"]
            edges = np.linspace(0, 301, out_len + 1).round().astype(int)
            waves = []
            for i in range(out_len):
                a, b = edges[i], edges[i + 1]
                if b <= a:
                    b = min(a + 1, 301)
                waves.append(float(waves301[a:b].mean()))

            # Input structure: (1, 128, 128) → flat list
            input_struct = sample_struct[0].tolist()  # 128x128 list

            # GT BGGR: (2, 2, 30) → nested list
            gt_bggr = [[sample_gt[0, 0, :].tolist(), sample_gt[0, 1, :].tolist()],
                       [sample_gt[1, 0, :].tolist(), sample_gt[1, 1, :].tolist()]]

            # Pred BGGR: (2, 2, 30) → nested list
            pred_bggr = [[sample_pred[0, 0, :].tolist(), sample_pred[0, 1, :].tolist()],
                         [sample_pred[1, 0, :].tolist(), sample_pred[1, 1, :].tolist()]]

            # 데이터 구성
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
                "data_stats": {
                    "train_size": trainer.data_stats["train_size"],
                    "val_size": trainer.data_stats["val_size"],
                    "total_size": trainer.data_stats["total_size"],
                    "batch_size": trainer.data_stats["batch_size"],
                },
                "sample": {
                    "input_struct": input_struct,
                    "gt_bggr": gt_bggr,
                    "pred_bggr": pred_bggr,
                    "waves": waves
                }
            }

            # Dashboard로 전송
            trainer.dashboard_server.push_update(data)

        except Exception as e:
            trainer.log(f"[DASHBOARD HOOK ERROR] {e}", also_console=False)
