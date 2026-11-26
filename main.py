"""Command-line entry point for the compact local fight screener."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector import analyze_video


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Path to an MP4/AVI or other OpenCV-readable video")
    parser.add_argument("--annotated", type=Path, help="Optional annotated MP4 output path")
    parser.add_argument("--storyboard", type=Path, help="Optional four-frame evidence JPEG path")
    args = parser.parse_args()
    result = analyze_video(args.video, output_path=args.annotated, snapshot_path=args.storyboard)
    result.pop("video_path", None)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
