#!/usr/bin/env python3
"""GVHMR runtime CLI entrypoint (library-first)."""

from __future__ import annotations

import argparse
from pathlib import Path

from hmr4d.infer_api import infer_video_to_pt


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GVHMR inference and output PT")
    parser.add_argument("--video", type=Path, required=True, help="Input MP4 path")
    parser.add_argument("--output_root", type=Path, required=True, help="Output root")
    parser.add_argument("-s", "--static_cam", action="store_true")
    parser.add_argument("--use_dpvo", action="store_true")
    parser.add_argument("--f_mm", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    pt_path = infer_video_to_pt(
        video_path=args.video,
        output_root=args.output_root,
        static_cam=args.static_cam,
        use_dpvo=args.use_dpvo,
        f_mm=args.f_mm,
    )
    print(str(pt_path))


if __name__ == "__main__":
    main()
