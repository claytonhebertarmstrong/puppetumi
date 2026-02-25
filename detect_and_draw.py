#!/usr/bin/env python3
"""Detect ArUco markers in an image and save an annotated copy."""

import argparse
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Input image path")
    parser.add_argument("--output", default=None, help="Output image path (default: <input>_detected.jpg)")
    parser.add_argument("--dict", default="DICT_4X4_50", help="ArUco dictionary (default: DICT_4X4_50)")
    parser.add_argument("--marker-size", type=float, default=0.03, help="Marker size in meters (default: 0.03)")
    args = parser.parse_args()

    dict_id = getattr(cv2.aruco, args.dict)
    aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)

    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)

    img = cv2.imread(args.image)
    if img is None:
        print(f"ERROR: Could not read {args.image}")
        return

    corners, ids, rejected = detector.detectMarkers(img)

    # Approximate camera matrix from image dimensions (no intrinsics needed for visualization)
    h, w = img.shape[:2]
    focal = max(w, h)
    K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    half = args.marker_size / 2.0
    obj_pts = np.array([
        [-half,  half, 0],
        [ half,  half, 0],
        [ half, -half, 0],
        [-half, -half, 0],
    ], dtype=np.float64)

    out = img.copy()
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(out, corners, ids)
        for i, marker_id in enumerate(ids):
            c = corners[i].squeeze().astype(np.float64)
            success, rvec, tvec = cv2.solvePnP(obj_pts, c, K, dist)
            if success:
                cv2.drawFrameAxes(out, K, dist, rvec, tvec, args.marker_size * 0.7, 3)
            cx, cy = c.mean(axis=0).astype(int)
            cv2.putText(out, f"ID:{marker_id[0]}", (cx - 30, cy - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        print(f"Detected {len(ids)} markers: {sorted(ids.flatten().tolist())}")
    else:
        print("No markers detected")

    output_path = args.output or args.image.rsplit(".", 1)[0] + "_detected.jpg"
    cv2.imwrite(output_path, out)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
