#!/usr/bin/env python3
"""Capture VirtualHome camera contact sheets for quick render debugging."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_PATH = REPO_ROOT / "virtualhome" / "virtualhome" / "simulation"
sys.path.insert(0, str(SIM_PATH))

from unity_simulator.comm_unity import UnityCommunication  # noqa: E402


def make_sheet(images, labels, cols, cell_w, cell_h, label_h):
    rows = math.ceil(len(images) / cols)
    sheet = Image.new("RGB", (cols * cell_w, rows * (cell_h + label_h)), (32, 32, 32))
    draw = ImageDraw.Draw(sheet)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % cols) * cell_w
        y = (index // cols) * (cell_h + label_h)
        draw.text((x + 4, y + 4), label, fill=(255, 255, 255))
        sheet.paste(image, (x, y + label_h))
    return sheet


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="8080")
    parser.add_argument("--scene", type=int, default=4)
    parser.add_argument("--out", default="debug/gaze2_modes_check")
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=90)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument(
        "--cameras",
        default="0,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,72",
    )
    parser.add_argument("--modes", default="normal,seg_inst,seg_class")
    args = parser.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    camera_ids = [int(value) for value in args.cameras.split(",") if value]
    modes = [value for value in args.modes.split(",") if value]

    comm = UnityCommunication(port=args.port)
    print("reset:", comm.reset(args.scene))
    print("camera_count:", comm.camera_count())

    for mode in modes:
        images = []
        labels = []
        for camera_id in camera_ids:
            ok, payload = comm.camera_image(
                [camera_id],
                mode=mode,
                image_width=args.width,
                image_height=args.height,
            )
            if ok and payload:
                image = Image.fromarray(payload[0]).convert("RGB")
            else:
                image = Image.new("RGB", (args.width, args.height), "black")
            overlay = ImageDraw.Draw(image)
            overlay.rectangle([0, 0, 36, 14], fill=(0, 0, 0))
            overlay.text((3, 1), str(camera_id), fill=(255, 255, 255))
            images.append(image)
            labels.append(f"{mode} cam {camera_id}")

        sheet = make_sheet(images, labels, args.cols, args.width, args.height, 26)
        path = out_dir / f"{mode}_sheet.png"
        sheet.save(path)
        print(path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
