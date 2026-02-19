"""
Standalone dashboard launcher.
Usage: python -m dashboard [--port 8501] [--config configs/default.yaml]

대시보드를 독립적으로 실행하여 브라우저에서 Train 버튼으로 학습을 시작할 수 있다.
"""
import argparse
import os
import signal
import sys
import threading
from pathlib import Path

# CR_recon 디렉토리를 sys.path에 추가 (from utils, trainer 등의 import를 위해)
_cr_recon_dir = str(Path(__file__).parent.parent)
if _cr_recon_dir not in sys.path:
    sys.path.insert(0, _cr_recon_dir)

from dashboard.server import DashboardServer
from dashboard.training_manager import TrainingManager


def main():
    parser = argparse.ArgumentParser(description="MetaSpec Dashboard (standalone)")
    parser.add_argument("--port", type=int, default=8501, help="Server port (default: 8501)")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Default config file path (e.g., configs/default.yaml)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  MetaSpec Dashboard (standalone mode)")
    print("=" * 60)

    server = DashboardServer(port=args.port, standalone=True)
    training_manager = TrainingManager(server, default_config=args.config)
    server.set_training_manager(training_manager)

    # config가 있으면 미리 model/loss 정보를 state에 설정 (Train 전에도 UI 표시)
    if args.config:
        try:
            from utils import load_config
            _cfg = load_config(args.config)
            server.state["model_name"] = _cfg["model"]["name"]
            server.state["model_params"] = _cfg["model"].get("params", {})
            server.state["loss_name"] = _cfg["loss"]["name"]
            server.state["loss_params"] = _cfg["loss"].get("params", {})
        except Exception as e:
            print(f"[WARNING] Could not pre-load config info: {e}")

    # Ctrl+C graceful shutdown
    _shutting_down = False

    def shutdown(sig, frame):
        nonlocal _shutting_down
        if _shutting_down:
            # 두 번째 Ctrl+C → 즉시 강제 종료
            print("\n[INFO] Force exit.")
            os._exit(1)
        _shutting_down = True
        print("\n[INFO] Shutting down...")

        def _cleanup():
            try:
                training_manager.stop_training()
            except Exception:
                pass
            try:
                server.stop()
            except Exception:
                pass
            # 정리 후 강제 종료 (uvicorn event loop가 걸릴 수 있으므로)
            os._exit(0)

        # 별도 스레드에서 정리 (signal handler에서 직접 하면 deadlock 가능)
        cleanup_thread = threading.Thread(target=_cleanup, daemon=True)
        cleanup_thread.start()
        # 3초 안에 정리 안되면 강제 종료
        cleanup_thread.join(timeout=3.0)
        if cleanup_thread.is_alive():
            print("[INFO] Cleanup timeout, force exit.")
        os._exit(0)

    signal.signal(signal.SIGINT, shutdown)

    import socket
    try:
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "localhost"

    print(f"  Local:     http://localhost:{args.port}")
    print(f"  Network:   http://{local_ip}:{args.port}")
    print("=" * 60)
    print("  Open the URL in your browser, then click 'Train' to start.")
    print("  Press Ctrl+C to stop the dashboard.")
    print("=" * 60 + "\n")

    # 블로킹 모드로 서버 시작 (메인 스레드에서 실행)
    server.start(blocking=True)


if __name__ == "__main__":
    main()
