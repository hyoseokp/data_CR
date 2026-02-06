"""
Dashboard Server: FastAPI + WebSocket
학습 상태를 실시간으로 브라우저에 전송한다.
"""
import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn

logger = logging.getLogger(__name__)


class DashboardServer:
    """
    학습 상태 데이터를 WebSocket으로 브라우저에 전송하는 대시보드 서버.
    별도 스레드에서 uvicorn 서버 구동.
    """

    def __init__(self, port: int = 8501):
        """
        port: 서버 포트 (기본 8501)
        """
        self.port = port
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
            "sample": None,
            "progress": {
                "stage": "",
                "epoch": 0,
                "total_epochs": 0,
                "batch": 0,
                "total_batches": 0,
                "current_loss": 0.0,
                "lr": 0.0
            }
        }

        # WebSocket 클라이언트 관리
        self.connected_clients: List[WebSocket] = []
        self.client_lock = asyncio.Lock()

        # 서버 스레드
        self.server_thread: Optional[threading.Thread] = None
        self.running = False
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    def _setup_routes(self):
        """FastAPI 라우트 설정."""

        @self.app.get("/api/status")
        async def get_status():
            """현재 학습 상태 반환."""
            return self.state

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint: 클라이언트와 연결하여 학습 데이터 전송."""
            await websocket.accept()
            # 클라이언트 목록에 추가
            async with self.client_lock:
                self.connected_clients.append(websocket)

            try:
                # 연결 직후 현재 상태 전송 (이전 데이터 복원)
                await websocket.send_text(json.dumps(self.state))

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
                return FileResponse(static_path)
            return {"message": "Dashboard frontend not found"}

    async def _broadcast(self, data: Dict[str, Any]):
        """모든 연결된 클라이언트에게 메시지 전송."""
        if not self.connected_clients:
            return

        message = json.dumps(data)
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
                host="127.0.0.1",
                port=self.port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(config)
            self.loop.run_until_complete(server.serve())
        except Exception as e:
            logger.error(f"Server error: {e}")
        finally:
            self.loop.close()

    def start(self):
        """서버 시작 (별도 스레드)."""
        if self.running:
            logger.warning("Server already running")
            return

        self.running = True
        self.server_thread = threading.Thread(target=self._run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"Dashboard server started on port {self.port}")

    def stop(self):
        """서버 종료."""
        if not self.running:
            return

        self.running = False
        if self.loop:
            # 모든 클라이언트 연결 종료
            try:
                asyncio.run_coroutine_threadsafe(self._close_all_clients(), self.loop)
            except Exception:
                pass

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
        return self.running and self.server_thread is not None and self.server_thread.is_alive()
