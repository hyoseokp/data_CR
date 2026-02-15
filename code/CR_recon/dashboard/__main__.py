"""
Standalone dashboard launcher.
Usage: python -m dashboard [--port 8501] [--config configs/default.yaml]

대시보드를 독립적으로 실행하여 브라우저에서 Train 버튼으로 학습을 시작할 수 있다.
"""
import argparse
import signal
import sys
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

    # Ctrl+C graceful shutdown
    def shutdown(sig, frame):
        print("\n[INFO] Shutting down...")
        training_manager.stop_training()
        server.stop()
        sys.exit(0)

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
