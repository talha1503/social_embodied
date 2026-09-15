#!/usr/bin/env python3
"""Query renderer_debug from a running VirtualHome Unity build."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = REPO_ROOT / "virtualhome" / "virtualhome" / "simulation"
sys.path.insert(0, str(SIM_PATH))

from unity_simulator.comm_unity import UnityCommunication  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="8080")
    parser.add_argument("--scene", type=int, default=4)
    parser.add_argument("--out", default="debug/gaze2_renderer_debug/renderer_debug.json")
    args = parser.parse_args()

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    comm = UnityCommunication(port=args.port)
    print("reset:", comm.reset(args.scene))
    ok, payload = comm.renderer_debug()
    print("renderer_debug:", ok)

    with out_path.open("w") as file:
        json.dump(payload, file, indent=2)
    print(out_path)

    if not ok or not isinstance(payload, dict):
        print(payload)
        return 1

    print("house:", payload.get("house"))
    print("renderer_count:", payload.get("renderer_count"))
    print("camera_count:", payload.get("camera_count"))
    for bucket, stats in sorted(payload.get("renderer_stats", {}).items()):
        print(
            f"{bucket}: total={stats.get('total')} "
            f"active={stats.get('activeInHierarchy')} "
            f"enabled={stats.get('enabled')} "
            f"active_enabled={stats.get('activeAndEnabled')} "
            f"layers={stats.get('layers')}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
