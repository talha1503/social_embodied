"""Launch and wait for the local macOS VirtualHome Unity app."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_APP = REPO_ROOT / "virtualhome/virtualhome/simulation/unity_simulator/macos_exec.2.2.4.app"


def wait_for_api(port: str, timeout_s: float) -> bool:
    sys.path.insert(0, str(REPO_ROOT / "virtualhome/virtualhome/simulation"))
    from unity_simulator.comm_unity import UnityCommunication

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            comm = UnityCommunication(port=port, timeout_wait=3)
            if comm.check_connection():
                return True
        except Exception:
            time.sleep(1.0)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch VirtualHome Unity on macOS.")
    parser.add_argument("--app", type=Path, default=DEFAULT_APP)
    parser.add_argument("--port", default="8080")
    parser.add_argument("--wait", type=float, default=30.0, help="Seconds to wait for the HTTP API.")
    parser.add_argument("--fresh", action="store_true", help="Quit existing VirtualHome app instances before launching.")
    args = parser.parse_args()

    if not args.app.exists():
        raise FileNotFoundError(args.app)

    if args.fresh:
        subprocess.run(["osascript", "-e", 'tell application "VirtualHome" to quit'], check=False)
        time.sleep(2.0)

    subprocess.run(
        [
            "/usr/bin/open",
            "-n",
            str(args.app),
            "--args",
            "-screen-fullscreen",
            "0",
            "-screen-quality",
            "4",
        ],
        check=True,
    )

    if wait_for_api(args.port, args.wait):
        print(f"VirtualHome API is responding on 127.0.0.1:{args.port}")
    else:
        raise TimeoutError(f"VirtualHome API did not respond on 127.0.0.1:{args.port} within {args.wait}s")


if __name__ == "__main__":
    main()
