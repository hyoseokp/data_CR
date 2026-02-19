"""
TrainingManager: 백그라운드 스레드에서 학습 lifecycle을 관리한다.
DashboardServer의 API 엔드포인트와 Trainer를 연결하는 중간 계층.
"""
import threading
import traceback
from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Any, List


class TrainingState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPING = "stopping"


class PreprocessState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingManager:
    """학습 lifecycle 관리: 시작, 중지, 상태 조회."""

    def __init__(self, server, default_config: Optional[str] = None):
        self.server = server
        self.default_config = default_config
        self._state = TrainingState.IDLE
        self._lock = threading.Lock()
        self._train_thread: Optional[threading.Thread] = None
        self._trainer = None
        self._error_message: Optional[str] = None
        self._stop_requested = False
        self._config_path: Optional[str] = None

        # 전처리 상태
        self._preprocess_state = PreprocessState.IDLE
        self._preprocess_lock = threading.Lock()
        self._preprocess_thread: Optional[threading.Thread] = None
        self._preprocess_error: Optional[str] = None
        self._preprocess_message: Optional[str] = None

    @property
    def state(self) -> TrainingState:
        with self._lock:
            return self._state

    @state.setter
    def state(self, value: TrainingState):
        with self._lock:
            self._state = value
        self._push_training_state()

    def _push_training_state(self):
        """현재 학습/전처리 제어 상태를 대시보드 클라이언트에 전송."""
        self.server.state["training_control"] = {
            "state": self._state.value,
            "error": self._error_message,
            "config_path": self._config_path,
        }
        self.server.state["preprocess_control"] = {
            "state": self._preprocess_state.value,
            "error": self._preprocess_error,
            "message": self._preprocess_message,
        }
        if self.server.is_running():
            try:
                self.server.push_update({})
            except Exception:
                pass

    def get_status(self) -> Dict[str, Any]:
        """현재 학습 상태 반환."""
        return {
            "state": self._state.value,
            "error": self._error_message,
            "config_path": self._config_path,
        }

    def list_configs(self) -> List[Dict[str, str]]:
        """configs/ 디렉토리의 YAML 파일 목록 반환."""
        configs_dir = Path(__file__).parent.parent / "configs"
        configs = []
        if configs_dir.is_dir():
            for f in sorted(configs_dir.glob("*.yaml")):
                configs.append({
                    "name": f.stem,
                    "path": str(f),
                    "filename": f.name,
                })
        return configs

    def get_checkpoint_info(self, config_path: str) -> Dict[str, Any]:
        """지정 config에 대한 체크포인트 존재 여부 및 정보 반환."""
        try:
            import sys
            import torch
            cr_recon_dir = str(Path(__file__).parent.parent)
            if cr_recon_dir not in sys.path:
                sys.path.insert(0, cr_recon_dir)
            from utils import load_config

            cfg = load_config(config_path)
            cfg_file = Path(config_path).resolve()
            cfg_dir = cfg_file.parent.parent  # CR_recon/
            # config에 output.dir이 있으면 사용, 없으면 기본 outputs/
            out_dir_name = cfg.get("output", {}).get("dir", "outputs")
            outputs_dir = cfg_dir / out_dir_name
            model_name = cfg["model"]["name"]
            total_epochs = cfg["training"]["epochs"]

            last_ckpt = outputs_dir / f"{model_name}_last.pt"
            best_ckpt = outputs_dir / f"{model_name}_best.pt"

            ckpt_path = None
            if last_ckpt.exists():
                ckpt_path = last_ckpt
            elif best_ckpt.exists():
                ckpt_path = best_ckpt

            if ckpt_path is None:
                return {"has_checkpoint": False}

            ckpt_data = torch.load(str(ckpt_path), map_location="cpu")
            ckpt_epoch = ckpt_data.get("epoch", 0)
            is_completed = (ckpt_epoch + 1 >= total_epochs)

            return {
                "has_checkpoint": True,
                "checkpoint_path": str(ckpt_path),
                "epoch": ckpt_epoch + 1,  # 1-indexed for display
                "total_epochs": total_epochs,
                "is_completed": is_completed,
                "best_val_loss": ckpt_data.get("best_val_loss"),
            }
        except Exception as e:
            return {"has_checkpoint": False, "error": str(e)}

    def start_training(
        self,
        config_path: str,
        resume_from: Optional[str] = None,
        init_weights: Optional[str] = None,
        skip_data_update: bool = True,
    ) -> Dict[str, Any]:
        """백그라운드 스레드에서 학습 시작."""
        if self._state in (TrainingState.RUNNING, TrainingState.STARTING):
            return {"ok": False, "error": "Training is already running"}

        config_file = Path(config_path)
        if not config_file.exists():
            return {"ok": False, "error": f"Config not found: {config_path}"}

        self._error_message = None
        self._config_path = str(config_file)
        self._stop_requested = False
        self.state = TrainingState.STARTING

        self._train_thread = threading.Thread(
            target=self._training_worker,
            args=(str(config_file), resume_from, init_weights, skip_data_update),
            daemon=True,
        )
        self._train_thread.start()
        return {"ok": True}

    def stop_training(self) -> Dict[str, Any]:
        """학습 graceful stop 요청 (현재 배치 완료 후 즉시 중지)."""
        if self._state not in (TrainingState.RUNNING, TrainingState.STARTING):
            return {"ok": False, "error": "No training is running"}
        self._stop_requested = True
        # trainer에 직접 플래그 설정 (배치 루프에서 즉시 체크)
        if self._trainer is not None:
            self._trainer._stop_requested = True
        self.state = TrainingState.STOPPING
        return {"ok": True, "message": "Stop requested, will stop after current batch"}

    def _training_worker(
        self,
        config_path: str,
        resume_from: Optional[str],
        init_weights: Optional[str],
        skip_data_update: bool,
    ):
        """실제 학습을 실행하는 백그라운드 스레드."""
        try:
            import sys
            # CR_recon 디렉토리를 sys.path에 추가
            cr_recon_dir = str(Path(__file__).parent.parent)
            if cr_recon_dir not in sys.path:
                sys.path.insert(0, cr_recon_dir)

            from utils import load_config
            from trainer import Trainer
            from train import validate_data_files, ensure_preprocessed_data

            cfg = load_config(config_path)

            # 데이터 검증
            cfg_file = Path(config_path).resolve()
            cfg_dir = cfg_file.parent.parent  # CR_recon/
            repo_root = cfg_dir.parent.parent
            data_dir = repo_root / "data_CR-main"

            if not skip_data_update:
                from train import ensure_latest_data
                if not validate_data_files(data_dir):
                    ensure_latest_data(data_dir)

            if not validate_data_files(data_dir):
                raise RuntimeError("Required data files missing or invalid")

            if not ensure_preprocessed_data(cfg_dir):
                raise RuntimeError("Data preprocessing failed")

            # 모델/손실 함수 정보를 대시보드에 즉시 전송
            self.server.state["model_name"] = cfg["model"]["name"]
            self.server.state["model_params"] = cfg["model"].get("params", {})
            self.server.state["loss_name"] = cfg["loss"]["name"]
            self.server.state["loss_params"] = cfg["loss"].get("params", {})
            if self.server.is_running():
                try:
                    self.server.push_update({})
                except Exception:
                    pass

            # 대시보드 활성화 (외부 서버 전달)
            cfg.setdefault("dashboard", {})["enabled"] = True

            # Trainer 생성 (외부 DashboardServer 전달)
            trainer = Trainer(cfg, dashboard_server=self.server)
            self._trainer = trainer

            # Stop 요청 시 trainer._stop_requested를 직접 설정 (배치 루프에서 즉시 체크)
            def stop_check_callback(**kwargs):
                if self._stop_requested:
                    trainer._stop_requested = True
                    raise KeyboardInterrupt("Stop requested from dashboard")

            trainer.add_callback(stop_check_callback)

            self.state = TrainingState.RUNNING

            # resume_from이 명시적으로 전달된 경우에만 resume
            # (Train 버튼 = 새 학습, Resume 버튼 = 체크포인트에서 이어서)
            trainer.train(resume_from=resume_from, init_weights=init_weights)

            self.state = TrainingState.COMPLETED

        except KeyboardInterrupt:
            self.state = TrainingState.IDLE
        except Exception as e:
            self._error_message = f"{type(e).__name__}: {e}"
            traceback.print_exc()
            self.state = TrainingState.FAILED
        finally:
            self._trainer = None

    # ===== 전처리 (Preprocess) =====

    @property
    def preprocess_state(self) -> PreprocessState:
        with self._preprocess_lock:
            return self._preprocess_state

    @preprocess_state.setter
    def preprocess_state(self, value: PreprocessState):
        with self._preprocess_lock:
            self._preprocess_state = value
        self._push_training_state()

    def get_preprocess_status(self) -> Dict[str, Any]:
        """현재 전처리 상태 반환."""
        cr_recon_dir = Path(__file__).parent.parent
        bayer_dir = cr_recon_dir / "dataset" / "bayer"
        required = [
            "struct_0.npy", "struct_1.npy",
            "bayer_0.npy", "bayer_1.npy",
            "bayer_rotated_0.npy", "bayer_rotated_1.npy",
        ]
        existing = [f for f in required if (bayer_dir / f).exists()]
        return {
            "state": self._preprocess_state.value,
            "error": self._preprocess_error,
            "message": self._preprocess_message,
            "data_ready": len(existing) == len(required),
            "files_found": len(existing),
            "files_total": len(required),
        }

    def start_preprocess(self) -> Dict[str, Any]:
        """백그라운드 스레드에서 데이터 전처리 시작."""
        if self._preprocess_state == PreprocessState.RUNNING:
            return {"ok": False, "error": "Preprocessing is already running"}
        if self._state in (TrainingState.RUNNING, TrainingState.STARTING):
            return {"ok": False, "error": "Cannot preprocess while training is running"}

        self._preprocess_error = None
        self._preprocess_message = "Starting..."
        self.preprocess_state = PreprocessState.RUNNING

        self._preprocess_thread = threading.Thread(
            target=self._preprocess_worker,
            daemon=True,
        )
        self._preprocess_thread.start()
        return {"ok": True}

    def _preprocess_worker(self):
        """실제 전처리를 실행하는 백그라운드 스레드."""
        try:
            import sys
            cr_recon_dir = str(Path(__file__).parent.parent)
            if cr_recon_dir not in sys.path:
                sys.path.insert(0, cr_recon_dir)

            import shutil
            from train import validate_data_files

            cfg_dir = Path(__file__).parent.parent  # CR_recon/
            repo_root = cfg_dir.parent.parent
            data_dir = repo_root / "data_CR-main"

            # 1) 원본 데이터 검증
            self._preprocess_message = "Validating raw data files..."
            self._push_training_state()
            if not validate_data_files(data_dir):
                raise RuntimeError(
                    "Required raw data files missing or invalid. "
                    "Place .npy files under data_CR-main/"
                )

            # 2) 기존 bayer 디렉토리 삭제 (재생성)
            bayer_dir = cfg_dir / "dataset" / "bayer"
            if bayer_dir.exists():
                self._preprocess_message = "Removing old preprocessed data..."
                self._push_training_state()
                shutil.rmtree(str(bayer_dir))

            # 3) preprocess_data.py 실행
            self._preprocess_message = "Running preprocessing (Spectra → Bayer)..."
            self._push_training_state()

            from preprocess_data import main as preprocess_main
            preprocess_main()

            # 4) 결과 검증
            required = [
                bayer_dir / "struct_0.npy", bayer_dir / "struct_1.npy",
                bayer_dir / "bayer_0.npy", bayer_dir / "bayer_1.npy",
                bayer_dir / "bayer_rotated_0.npy", bayer_dir / "bayer_rotated_1.npy",
            ]
            if not all(f.exists() for f in required):
                raise RuntimeError("Preprocessing completed but output files are missing")

            self._preprocess_message = "Preprocessing completed successfully"
            self.preprocess_state = PreprocessState.COMPLETED

        except Exception as e:
            self._preprocess_error = f"{type(e).__name__}: {e}"
            self._preprocess_message = None
            traceback.print_exc()
            self.preprocess_state = PreprocessState.FAILED
        finally:
            self._preprocess_thread = None
