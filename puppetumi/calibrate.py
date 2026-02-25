#!/usr/bin/env python3
"""Fisheye camera calibration using a ChArUco board.

Subcommands:
    generate-board  Render a printable ChArUco board PNG.
    calibrate       Calibrate fisheye intrinsics from a set of images or a video.

Examples:
    python -m puppetumi.calibrate generate-board --output charuco_board.png
    python -m puppetumi.calibrate calibrate \
        --input /path/to/calibration_images/ \
        --output camera_intrinsics.json
    python -m puppetumi.calibrate calibrate \
        --input /path/to/calibration_video.mp4 \
        --output camera_intrinsics.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ── ChArUco board helpers ──────────────────────────────────────────

CHARUCO_DICT = cv2.aruco.DICT_6X6_50
CHARUCO_GRID = (12, 8)
CHARUCO_SQUARE_MM = 50.0
CHARUCO_MARKER_MM = 30.0


def get_charuco_board():
    aruco_dict = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT)
    return cv2.aruco.CharucoBoard(
        size=CHARUCO_GRID,
        squareLength=CHARUCO_SQUARE_MM / 1000,
        markerLength=CHARUCO_MARKER_MM / 1000,
        dictionary=aruco_dict,
    )


def draw_charuco_board(board, dpi=300, padding_mm=15):
    grid_size = np.array(board.getChessboardSize())
    square_mm = board.getSquareLength() * 1000
    mm_per_inch = 25.4
    board_px = ((grid_size * square_mm + padding_mm * 2) / mm_per_inch * dpi).round().astype(int)
    padding_px = int(padding_mm / mm_per_inch * dpi)
    return board.generateImage(outSize=board_px, marginSize=padding_px)


# ── Frame loading ──────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def load_frames(input_path: str):
    """Load frames from a directory of images or a video file.

    Returns (frames, width, height).
    """
    p = Path(input_path)

    if p.is_dir():
        paths = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if not paths:
            print(f"ERROR: No images found in {p}", file=sys.stderr)
            sys.exit(1)
        frames = []
        for img_path in paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                frames.append(img)
        h, w = frames[0].shape[:2]
        return frames, w, h

    if p.suffix.lower() in VIDEO_EXTS:
        return _decode_video(str(p))

    print(f"ERROR: {p} is not a directory or a supported video file", file=sys.stderr)
    sys.exit(1)


def _decode_video(video_path: str):
    probe = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "csv=p=0",
            video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    vid_w, vid_h = (int(x) for x in probe.stdout.strip().split(","))

    proc = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "error",
            "-i", video_path,
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-v", "error", "-",
        ],
        stdout=subprocess.PIPE,
    )
    frame_size = vid_w * vid_h * 3
    frames = []
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) != frame_size:
            break
        frames.append(np.frombuffer(raw, dtype=np.uint8).reshape(vid_h, vid_w, 3))
    proc.stdout.close()
    proc.wait()
    return frames, vid_w, vid_h


# ── Calibration ────────────────────────────────────────────────────

def calibrate(frames, image_size, skip_frames=1, max_frames=80):
    board = get_charuco_board()
    detector = cv2.aruco.CharucoDetector(board)

    all_corners, all_ids = [], []
    selected = frames[::skip_frames]
    print(f"Processing {len(selected)} frames (skip={skip_frames})")

    for i, frame in enumerate(selected):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
        if charuco_ids is not None and len(charuco_ids) >= 8:
            all_corners.append(charuco_corners)
            all_ids.append(charuco_ids)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(selected)}, {len(all_corners)} usable")

    print(f"Usable frames: {len(all_corners)} / {len(selected)}")
    if len(all_corners) < 5:
        print("ERROR: Need at least 5 usable frames.", file=sys.stderr)
        sys.exit(1)

    if len(all_corners) > max_frames:
        step = len(all_corners) // max_frames
        all_corners = all_corners[::step][:max_frames]
        all_ids = all_ids[::step][:max_frames]
        print(f"Subsampled to {len(all_corners)} frames")

    board_obj_pts = board.getChessboardCorners()
    obj_points, img_points = [], []
    for corners, ids in zip(all_corners, all_ids):
        obj_pts = np.array([board_obj_pts[i[0]] for i in ids], dtype=np.float64)
        obj_points.append(obj_pts.reshape(1, -1, 3))
        img_points.append(corners.reshape(1, -1, 2).astype(np.float64))

    K = np.eye(3, dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_FIX_SKEW

    while len(obj_points) >= 5:
        try:
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                obj_points, img_points, image_size, K, D,
                flags=flags,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
            )
            break
        except cv2.error:
            min_idx = min(range(len(obj_points)), key=lambda i: obj_points[i].shape[1])
            obj_points.pop(min_idx)
            img_points.pop(min_idx)
            print(f"  Removed problematic frame, {len(obj_points)} remaining")
    else:
        print("ERROR: Calibration failed.", file=sys.stderr)
        sys.exit(1)

    print(f"Reprojection error (RMS): {rms:.4f}")
    print(f"K:\n{K}")
    print(f"D: {D.flatten()}")
    return K, D, rms


# ── CLI ────────────────────────────────────────────────────────────

def cmd_generate_board(args):
    board = get_charuco_board()
    img = draw_charuco_board(board, dpi=args.dpi)
    cv2.imwrite(args.output, img)
    print(f"Board saved to {args.output} ({img.shape[1]}x{img.shape[0]} px, {args.dpi} DPI)")


def cmd_calibrate(args):
    frames, w, h = load_frames(args.input)
    print(f"Loaded {len(frames)} frames ({w}x{h})")
    K, D, rms = calibrate(frames, (w, h), skip_frames=args.skip_frames)

    result = {
        "K": K.tolist(),
        "D": D.flatten().tolist(),
        "DIM": [int(w), int(h)],
        "rms_reproj_error": float(rms),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Intrinsics saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description="Fisheye camera calibration")
    sub = parser.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("generate-board", help="Render printable ChArUco board PNG")
    bp.add_argument("--output", default="charuco_board.png")
    bp.add_argument("--dpi", type=int, default=300)

    cp = sub.add_parser("calibrate", help="Calibrate fisheye intrinsics")
    cp.add_argument("--input", required=True, help="Directory of images or path to video")
    cp.add_argument("--output", default="camera_intrinsics.json")
    cp.add_argument("--skip-frames", type=int, default=1)

    args = parser.parse_args()
    if args.command == "generate-board":
        cmd_generate_board(args)
    else:
        cmd_calibrate(args)


if __name__ == "__main__":
    main()
