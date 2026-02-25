#!/usr/bin/env python3
"""Solve for a reference point's 6-DoF pose in camera frame using ArUco markers.

Given:
    - A marker config defining ArUco marker poses relative to a reference point
    - Camera intrinsics (fisheye model: K, D)
    - A directory of images captured from a static camera

For each image:
    1. Detect ArUco markers
    2. Undistort the detected 2D corners (fisheye → pinhole)
    3. Pool all 3D marker-corner positions (in reference frame) and their
       corresponding 2D detections
    4. Solve PnP (RANSAC + Levenberg-Marquardt refinement) for the
       reference-frame → camera-frame transform
    5. Output the reference point's position and orientation in camera frame

Usage:
    python -m puppetumi.solve \
        --images /path/to/captured_images/ \
        --marker-config marker_config.json \
        --intrinsics camera_intrinsics.json \
        --output results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# ── Config loading ─────────────────────────────────────────────────

ARUCO_DICT_MAP = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
}


def load_marker_config(path: str) -> dict:
    """Load marker config JSON.

    Expected format:
        {
            "aruco_dictionary": "DICT_4X4_50",
            "marker_size_m": 0.03,
            "markers": {
                "0": {
                    "position": [x, y, z],
                    "quaternion_wxyz": [w, x, y, z]
                },
                ...
            }
        }

    position: marker center in the reference frame (meters).
    quaternion_wxyz: rotation from marker-local frame to reference frame.
    Marker-local frame: Z out of marker face, corners at ±half_size in X/Y.
    """
    with open(path) as f:
        cfg = json.load(f)

    dict_name = cfg["aruco_dictionary"]
    if dict_name not in ARUCO_DICT_MAP:
        print(f"ERROR: Unknown ArUco dictionary '{dict_name}'", file=sys.stderr)
        sys.exit(1)

    return cfg


def load_intrinsics(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load fisheye intrinsics (K, D) from JSON."""
    with open(path) as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(4, 1)
    return K, D


# ── Geometry helpers ───────────────────────────────────────────────

def marker_corners_in_reference_frame(
    position: List[float],
    quaternion_wxyz: List[float],
    half_size: float,
) -> np.ndarray:
    """Compute the 4 marker corners in the reference frame.

    Marker-local corners (OpenCV convention, CCW from top-left):
        [-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]

    Returns: (4, 3) array of 3D points in reference frame.
    """
    local_corners = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0],
    ], dtype=np.float64)

    w, x, y, z = quaternion_wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()  # scipy uses xyzw
    t = np.array(position, dtype=np.float64)

    return (R @ local_corners.T).T + t


def build_object_points(cfg: dict) -> Dict[int, np.ndarray]:
    """Pre-compute 3D corner positions in reference frame for every marker.

    Returns: {marker_id: (4, 3) ndarray}
    """
    half = cfg["marker_size_m"] / 2.0
    result = {}
    for id_str, marker in cfg["markers"].items():
        mid = int(id_str)
        corners_3d = marker_corners_in_reference_frame(
            marker["position"],
            marker["quaternion_wxyz"],
            half,
        )
        result[mid] = corners_3d
    return result


# ── Detection ──────────────────────────────────────────────────────

def detect_markers(
    img: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
) -> Dict[int, np.ndarray]:
    """Detect ArUco markers in an image.

    Returns: {marker_id: (4, 2) corner array}
    """
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    detector = cv2.aruco.ArucoDetector(aruco_dict, params)
    corners, ids, _ = detector.detectMarkers(img)
    if ids is None or len(corners) == 0:
        return {}
    return {int(ids[i][0]): corners[i].squeeze() for i in range(len(ids))}


# ── Solver ─────────────────────────────────────────────────────────

def solve_reference_point(
    img: np.ndarray,
    aruco_dict: cv2.aruco.Dictionary,
    object_points_map: Dict[int, np.ndarray],
    K: np.ndarray,
    D: np.ndarray,
) -> Optional[dict]:
    """Detect markers and solve for the reference point pose in camera frame.

    Returns dict with rvec, tvec, num_markers, reprojection_error, or None.
    """
    detections = detect_markers(img, aruco_dict)

    # Filter to markers we have config for
    known_ids = set(object_points_map.keys()) & set(detections.keys())
    if len(known_ids) < 1:
        return None

    # Pool all 3D object points and 2D image points
    obj_pts_list = []
    img_pts_list = []
    for mid in known_ids:
        obj_pts_list.append(object_points_map[mid])           # (4, 3)
        img_pts_list.append(detections[mid].astype(np.float64))  # (4, 2)

    obj_pts = np.vstack(obj_pts_list)  # (N*4, 3)
    img_pts = np.vstack(img_pts_list)  # (N*4, 2)

    # Undistort image points (fisheye → ideal pinhole)
    img_pts_undist = cv2.fisheye.undistortPoints(
        img_pts.reshape(-1, 1, 2), K, D, P=K
    ).reshape(-1, 2)

    # Solve PnP (zero distortion since we already undistorted)
    no_dist = np.zeros((1, 5), dtype=np.float64)
    n_points = obj_pts.shape[0]

    if n_points >= 6:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj_pts, img_pts_undist, K, no_dist,
            iterationsCount=200,
            reprojectionError=5.0,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
        # Refine with Levenberg-Marquardt using inliers
        if inliers is not None and len(inliers) >= 4:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj_pts[inliers.flatten()],
                img_pts_undist[inliers.flatten()],
                K, no_dist, rvec, tvec,
            )
    elif n_points >= 4:
        success, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts_undist, K, no_dist,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return None
    else:
        return None

    # Compute reprojection error
    projected, _ = cv2.projectPoints(obj_pts, rvec, tvec, K, no_dist)
    reproj_err = np.sqrt(np.mean((projected.reshape(-1, 2) - img_pts_undist) ** 2))

    return {
        "rvec": rvec.squeeze(),
        "tvec": tvec.squeeze(),
        "num_markers_detected": len(known_ids),
        "num_corners_used": n_points,
        "reprojection_error_px": float(reproj_err),
    }


def transform_to_reference_point(rvec, tvec, ref_cfg):
    """Transform from ArUco 1 frame to the reference point (fingertip).

    PnP gives us T_cam_aruco1 (ArUco 1 in camera frame).
    ref_cfg gives us t_aruco1_ref (reference point in ArUco 1 frame).
    We compute: p_cam_ref = R_cam_aruco1 @ t_aruco1_ref + t_cam_aruco1
    """
    R_cam_aruco1, _ = cv2.Rodrigues(rvec)
    t_aruco1_ref = np.array(ref_cfg["position"], dtype=np.float64)

    t_cam_ref = R_cam_aruco1 @ t_aruco1_ref + tvec

    # Reference point orientation in camera frame
    w, x, y, z = ref_cfg["quaternion_wxyz"]
    R_aruco1_ref = Rotation.from_quat([x, y, z, w]).as_matrix()
    R_cam_ref = R_cam_aruco1 @ R_aruco1_ref
    rvec_ref, _ = cv2.Rodrigues(R_cam_ref)

    return rvec_ref.squeeze(), t_cam_ref


# ── Main pipeline ─────────────────────────────────────────────────

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def run(args):
    cfg = load_marker_config(args.marker_config)
    K, D = load_intrinsics(args.intrinsics)
    aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_MAP[cfg["aruco_dictionary"]])
    obj_pts_map = build_object_points(cfg)

    ref_cfg = cfg.get("reference_point")

    print(f"Loaded {len(cfg['markers'])} markers from config")
    print(f"ArUco dictionary: {cfg['aruco_dictionary']}")
    print(f"Marker size: {cfg['marker_size_m']*100:.1f} cm")
    if ref_cfg:
        print(f"Reference point: {ref_cfg.get('description', 'custom')} at {ref_cfg['position']}")
    else:
        print("Reference point: ArUco 1 origin (no reference_point in config)")

    images_dir = Path(args.images)
    image_paths = sorted(f for f in images_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        print(f"ERROR: No images found in {images_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Processing {len(image_paths)} images...")

    results = []
    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  WARN: Could not read {img_path.name}, skipping")
            continue

        result = solve_reference_point(img, aruco_dict, obj_pts_map, K, D)
        if result is None:
            print(f"  [{i+1}/{len(image_paths)}] {img_path.name}: no solution (not enough markers)")
            results.append({"image": img_path.name, "solved": False})
            continue

        rvec_out = result["rvec"]
        tvec_out = result["tvec"]

        # Transform to fingertip / reference point if configured
        if ref_cfg:
            rvec_out, tvec_out = transform_to_reference_point(
                result["rvec"], result["tvec"], ref_cfg
            )

        print(
            f"  [{i+1}/{len(image_paths)}] {img_path.name}: "
            f"pos=({tvec_out[0]:+.4f}, {tvec_out[1]:+.4f}, {tvec_out[2]:+.4f}) m, "
            f"{result['num_markers_detected']} markers, "
            f"reproj={result['reprojection_error_px']:.2f} px"
        )
        results.append({
            "image": img_path.name,
            "solved": True,
            "reference_point_position_m": tvec_out.tolist(),
            "reference_point_rvec": rvec_out.tolist(),
            "num_markers_detected": result["num_markers_detected"],
            "num_corners_used": result["num_corners_used"],
            "reprojection_error_px": result["reprojection_error_px"],
        })

    solved = sum(1 for r in results if r.get("solved"))
    print(f"\nSolved {solved}/{len(results)} frames")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"frames": results}, f, indent=2)
    print(f"Results saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Solve for reference point pose in camera frame from ArUco markers"
    )
    parser.add_argument("--images", required=True, help="Directory of captured images")
    parser.add_argument("--marker-config", required=True, help="Marker config JSON")
    parser.add_argument("--intrinsics", required=True, help="Camera intrinsics JSON")
    parser.add_argument("--output", default="results.json", help="Output JSON path")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
