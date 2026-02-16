"""
Dashboard Server: FastAPI + WebSocket
학습 상태를 실시간으로 브라우저에 전송한다.
"""
import asyncio
import json
import logging
import math
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TrainStartRequest(BaseModel):
    """학습 시작 요청 바디."""
    config_path: str
    resume_from: Optional[str] = None
    init_weights: Optional[str] = None
    skip_data_update: bool = True


class LossWeightsRequest(BaseModel):
    """Loss weight 업데이트 요청."""
    w_mse: Optional[float] = None
    w_rel: Optional[float] = None
    w_grad: Optional[float] = None

def _json_sanitize(obj: Any) -> Any:
    """
    Make payload safe for browser-side JSON.parse:
    - Replace NaN/Inf floats with None (-> null)
    - Recurse through dict/list/tuple
    """
    if obj is None:
        return None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    return str(obj)


class DashboardServer:
    """
    학습 상태 데이터를 WebSocket으로 브라우저에 전송하는 대시보드 서버.
    별도 스레드에서 uvicorn 서버 구동.
    """

    def __init__(self, port: int = 8501, standalone: bool = False):
        """
        port: 서버 포트 (기본 8501)
        standalone: True이면 독립 실행 모드 (학습 제어 API 활성화)
        """
        self.port = port
        self.standalone = standalone
        self._training_manager = None
        self.app = FastAPI(title="MetaSpec Dashboard")
        self._setup_routes()

        # 상태 관리
        self.state = {
            "epoch": 0,
            "total_epochs": 0,
            "lr": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "best_val": float("inf"),
            "train_losses": [],
            "val_losses": [],
            "samples": {
                "val": None,
                "train": None,
                "waves": []
            },
            "progress": {
                "stage": "",
                "epoch": 0,
                "total_epochs": 0,
                "batch": 0,
                "total_batches": 0,
                "current_loss": 0.0,
                "lr": 0.0
            },
            "training_control": {
                "state": "idle",
                "error": None,
                "config_path": None,
            },
            "preprocess_control": {
                "state": "idle",
                "error": None,
                "message": None,
            }
        }

        # WebSocket 클라이언트 관리
        self.connected_clients: List[WebSocket] = []
        self.client_lock = asyncio.Lock()

        # 서버 스레드
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self._uvicorn_server = None

    def _setup_routes(self):
        """FastAPI 라우트 설정."""

        @self.app.get("/api/status")
        async def get_status():
            """현재 학습 상태 반환."""
            return _json_sanitize(self.state)

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint: 클라이언트와 연결하여 학습 데이터 전송."""
            await websocket.accept()
            # 클라이언트 목록에 추가
            async with self.client_lock:
                self.connected_clients.append(websocket)

            try:
                # 연결 직후 현재 상태 전송 (이전 데이터 복원)
                await websocket.send_text(json.dumps(_json_sanitize(self.state), allow_nan=False))

                # 연결 유지 (클라이언트로부터 메시지 대기)
                while True:
                    _ = await websocket.receive_text()
            except Exception as e:
                logger.debug(f"WebSocket error: {e}")
            finally:
                # 클라이언트 목록에서 제거
                async with self.client_lock:
                    if websocket in self.connected_clients:
                        self.connected_clients.remove(websocket)

        @self.app.get("/")
        async def root():
            """index.html 서빙."""
            static_path = Path(__file__).parent / "static" / "index.html"
            if static_path.exists():
                # Avoid stale frontend during rapid iteration (browser cache can mask updates).
                return FileResponse(
                    static_path,
                    headers={
                        "Cache-Control": "no-store, max-age=0",
                        "Pragma": "no-cache",
                        "Expires": "0",
                    },
                )
            return {"message": "Dashboard frontend not found"}

        # ===== Training Control API (standalone mode) =====

        @self.app.get("/api/standalone")
        async def is_standalone():
            """standalone 모드 여부 확인."""
            return {"standalone": self.standalone}

        @self.app.get("/api/train/configs")
        async def list_configs():
            """사용 가능한 config 파일 목록."""
            if not self._training_manager:
                return {"configs": []}
            return {"configs": self._training_manager.list_configs()}

        @self.app.get("/api/train/status")
        async def get_train_status():
            """현재 학습 상태."""
            if not self._training_manager:
                return {"state": "unavailable", "error": "Not in standalone mode"}
            return self._training_manager.get_status()

        @self.app.post("/api/train/start")
        async def start_training(req: TrainStartRequest):
            """학습 시작."""
            if not self._training_manager:
                return {"ok": False, "error": "Not in standalone mode"}
            return self._training_manager.start_training(
                config_path=req.config_path,
                resume_from=req.resume_from,
                init_weights=req.init_weights,
                skip_data_update=req.skip_data_update,
            )

        @self.app.get("/api/train/checkpoint")
        async def get_checkpoint_info(config_path: str = ""):
            """선택된 config에 대한 체크포인트 정보."""
            if not self._training_manager:
                return {"has_checkpoint": False}
            if not config_path:
                return {"has_checkpoint": False, "error": "No config path"}
            return self._training_manager.get_checkpoint_info(config_path)

        @self.app.post("/api/train/stop")
        async def stop_training():
            """학습 중지 요청."""
            if not self._training_manager:
                return {"ok": False, "error": "Not in standalone mode"}
            return self._training_manager.stop_training()

        # ===== Loss Weights API =====

        @self.app.post("/api/train/loss_weights")
        async def update_loss_weights(req: LossWeightsRequest):
            """학습 중 loss weight를 실시간으로 변경."""
            if not self._training_manager:
                return {"ok": False, "error": "Not in standalone mode"}
            trainer = self._training_manager._trainer
            if trainer is None or not hasattr(trainer, 'loss_fn'):
                return {"ok": False, "error": "No active training session"}
            loss_fn = trainer.loss_fn
            updated = {}
            if req.w_mse is not None and hasattr(loss_fn, 'w_mse'):
                loss_fn.w_mse = float(req.w_mse)
                updated['w_mse'] = loss_fn.w_mse
            if req.w_rel is not None and hasattr(loss_fn, 'w_rel'):
                loss_fn.w_rel = float(req.w_rel)
                updated['w_rel'] = loss_fn.w_rel
            if req.w_grad is not None and hasattr(loss_fn, 'w_grad'):
                loss_fn.w_grad = float(req.w_grad)
                updated['w_grad'] = loss_fn.w_grad
            # 대시보드 state에도 반영
            if 'loss_params' in self.state:
                self.state['loss_params'].update(updated)
            # live_weights도 업데이트 (dashboard breakdown용)
            if 'live_weights' not in self.state:
                self.state['live_weights'] = {}
            self.state['live_weights'].update(updated)
            return {"ok": True, "updated": updated}

        @self.app.get("/api/train/loss_weights")
        async def get_loss_weights():
            """현재 loss weight 반환."""
            trainer = getattr(self._training_manager, '_trainer', None) if self._training_manager else None
            if trainer is None or not hasattr(trainer, 'loss_fn'):
                return {"ok": False, "error": "No active training session"}
            loss_fn = trainer.loss_fn
            return {
                "ok": True,
                "weights": {
                    "w_mse": getattr(loss_fn, 'w_mse', None),
                    "w_rel": getattr(loss_fn, 'w_rel', None),
                    "w_grad": getattr(loss_fn, 'w_grad', None),
                }
            }

        # ===== Preprocess API =====

        @self.app.get("/api/preprocess/status")
        async def get_preprocess_status():
            """현재 전처리 상태."""
            if not self._training_manager:
                return {"state": "unavailable", "error": "Not in standalone mode"}
            return self._training_manager.get_preprocess_status()

        @self.app.post("/api/preprocess/start")
        async def start_preprocess():
            """데이터 전처리 시작."""
            if not self._training_manager:
                return {"ok": False, "error": "Not in standalone mode"}
            try:
                return self._training_manager.start_preprocess()
            except Exception as e:
                logger.error(f"start_preprocess error: {e}")
                return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    async def _broadcast(self, data: Dict[str, Any]):
        """모든 연결된 클라이언트에게 메시지 전송."""
        if not self.connected_clients:
            return

        message = json.dumps(_json_sanitize(data), allow_nan=False)
        disconnected = []

        async with self.client_lock:
            for client in self.connected_clients:
                try:
                    await client.send_text(message)
                except Exception as e:
                    logger.debug(f"Failed to send message to client: {e}")
                    disconnected.append(client)

            # 연결이 끊긴 클라이언트 제거
            for client in disconnected:
                if client in self.connected_clients:
                    self.connected_clients.remove(client)

    def push_update(self, data: Dict[str, Any]):
        """
        학습 데이터 push (trainer callback에서 호출).

        Args:
            data: 학습 상태 dict
                {
                    "epoch": int,
                    "total_epochs": int,
                    "lr": float,
                    "train_loss": float,
                    "val_loss": float,
                    "best_val": float,
                    "train_losses": list,
                    "val_losses": list,
                    "sample": dict (optional)
                }
        """
        if not self.running or self.loop is None:
            return

        # 상태 업데이트
        self.state.update(data)

        # WebSocket으로 전송 (비동기)
        try:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(self.state),
                self.loop
            )
        except Exception as e:
            logger.debug(f"Failed to push update: {e}")

    def push_progress(self, progress: Dict[str, Any]):
        """
        배치 진행 상황 push (trainer에서 배치마다 호출).

        Args:
            progress: 진행 상황 dict
                {
                    "stage": "train" or "val",
                    "epoch": int,
                    "total_epochs": int,
                    "batch": int,
                    "total_batches": int,
                    "current_loss": float,
                    "lr": float (optional)
                }
        """
        if not self.running or self.loop is None:
            return

        # 진행 상황 업데이트
        self.state["progress"].update(progress)

        # WebSocket으로 전송 (비동기)
        try:
            asyncio.run_coroutine_threadsafe(
                self._broadcast(self.state),
                self.loop
            )
        except Exception as e:
            logger.debug(f"Failed to push progress: {e}")

    def _run_server(self):
        """uvicorn 서버 실행 (별도 스레드에서 호출)."""
        # 새 event loop 생성
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        try:
            config = uvicorn.Config(
                app=self.app,
                host="0.0.0.0",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            self._uvicorn_server = uvicorn.Server(config)
            self.loop.run_until_complete(self._uvicorn_server.serve())
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.loop.close()

    def start(self, blocking: bool = False):
        """
        서버 시작.
        blocking=False: 별도 스레드에서 실행 (기존 동작, CLI 학습 모드)
        blocking=True: 현재 스레드에서 실행 (standalone 대시보드 모드)
        """
        if self.running:
            logger.warning("Server already running")
            return

        self.running = True

        if blocking:
            # standalone 모드: 메인 스레드에서 직접 실행 (블로킹)
            self._run_server()
        else:
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()

            # 서버 시작 대기 (최대 5초)
            import time
            for _ in range(50):
                if self.loop is not None:
                    break
                time.sleep(0.1)

            logger.info(f"Dashboard server started on port {self.port}")

    def stop(self):
        """서버 종료."""
        if not self.running:
            return

        self.running = False

        # uvicorn 서버에 종료 시그널 전송
        if hasattr(self, '_uvicorn_server') and self._uvicorn_server:
            self._uvicorn_server.should_exit = True

        if self.server_thread:
            self.server_thread.join(timeout=5.0)

        logger.info("Dashboard server stopped")

    async def _close_all_clients(self):
        """모든 WebSocket 클라이언트 연결 종료."""
        async with self.client_lock:
            for client in self.connected_clients:
                try:
                    await client.close()
                except Exception:
                    pass
            self.connected_clients.clear()

    def is_running(self) -> bool:
        """서버 실행 상태."""
        if not self.running:
            return False
        # standalone(blocking) 모드에서는 server_thread가 없음
        if self.server_thread is not None:
            return self.server_thread.is_alive()
        # blocking 모드: loop가 존재하면 실행 중
        return self.loop is not None

    def set_training_manager(self, manager):
        """TrainingManager 연결 (standalone 모드)."""
        self._training_manager = manager

    def reset_state(self):
        """상태 초기화 (새 훈련 시작 시 호출)."""
        # training_control, preprocess_control, model/loss 정보 보존
        tc = self.state.get("training_control", {
            "state": "idle",
            "error": None,
            "config_path": None,
        })
        pc = self.state.get("preprocess_control", {
            "state": "idle",
            "error": None,
            "message": None,
        })
        mn = self.state.get("model_name")
        mp = self.state.get("model_params")
        ln = self.state.get("loss_name")
        lp = self.state.get("loss_params")
        self.state = {
            "epoch": 0,
            "total_epochs": 0,
            "lr": 0.0,
            "train_loss": 0.0,
            "val_loss": 0.0,
            "best_val": float("inf"),
            "train_losses": [],
            "val_losses": [],
            "samples": {
                "val": None,
                "train": None,
                "waves": []
            },
            "progress": {
                "stage": "",
                "epoch": 0,
                "total_epochs": 0,
                "batch": 0,
                "total_batches": 0,
                "current_loss": 0.0,
                "lr": 0.0
            },
            "training_control": tc,
            "preprocess_control": pc,
        }
        if mn:
            self.state["model_name"] = mn
            self.state["model_params"] = mp
        if ln:
            self.state["loss_name"] = ln
            self.state["loss_params"] = lp
