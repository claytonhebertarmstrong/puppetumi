#!/usr/bin/env python3
"""Record video from a USB camera for calibration.

Usage:
    python -m puppetumi.record_video --output calibration_video.mp4
    python -m puppetumi.record_video --camera 1 --output calibration_video.mp4

Press Ctrl+C to stop recording.
"""

import argparse
import signal
import time

import cv2


def main():
    parser = argparse.ArgumentParser(description="Record video from USB camera")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--output", default="calibration_video.mp4", help="Output video path")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Could not open camera {args.camera}")
        return

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        print(f"ERROR: Could not open video writer for {args.output}")
        cap.release()
        return

    print(f"Recording {w}x{h} @ {fps:.0f} FPS -> {args.output}")
    print("Press Ctrl+C to stop.")

    stop = False

    def on_signal(sig, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)

    frame_count = 0
    start = time.time()

    while not stop:
        ret, frame = cap.read()
        if not ret:
            continue
        writer.write(frame)
        frame_count += 1
        if frame_count % 30 == 0:
            elapsed = time.time() - start
            print(f"  {frame_count} frames ({elapsed:.1f}s)", end="\r")

    elapsed = time.time() - start
    cap.release()
    writer.release()
    print(f"\nSaved {frame_count} frames ({elapsed:.1f}s) to {args.output}")


if __name__ == "__main__":
    main()
