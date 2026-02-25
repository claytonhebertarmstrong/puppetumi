#!/usr/bin/env python3
"""Visualize marker config: draw each ArUco marker as a square with axes in 3D."""

import json
import sys

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation


def load_config(path):
    with open(path) as f:
        return json.load(f)


def marker_corners_3d(position, quaternion_wxyz, half_size):
    local = np.array([
        [-half_size,  half_size, 0],
        [ half_size,  half_size, 0],
        [ half_size, -half_size, 0],
        [-half_size, -half_size, 0],
    ])
    w, x, y, z = quaternion_wxyz
    R = Rotation.from_quat([x, y, z, w]).as_matrix()
    t = np.array(position)
    return (R @ local.T).T + t, R, t


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "marker_config.json"
    cfg = load_config(config_path)
    half = cfg["marker_size_m"] / 2.0

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    colors = {
        "1": "#4CAF50",
        "2": "#2196F3",
        "3": "#FF9800",
        "4": "#E91E63",
    }

    axis_len = cfg["marker_size_m"] * 0.6

    for id_str, marker in cfg["markers"].items():
        corners, R, t = marker_corners_3d(marker["position"], marker["quaternion_wxyz"], half)

        # Draw marker face
        verts = [corners.tolist()]
        poly = Poly3DCollection(verts, alpha=0.3, facecolor=colors.get(id_str, "gray"),
                                edgecolor="black", linewidth=1.5)
        ax.add_collection3d(poly)

        # Draw axes: R=X, G=Y, B=Z
        for col, color in [(0, "red"), (1, "green"), (2, "blue")]:
            axis_dir = R[:, col] * axis_len
            ax.quiver(*t, *axis_dir, color=color, arrow_length_ratio=0.15, linewidth=2)

        # Label
        ax.text(*t, f"  ID:{id_str}", fontsize=11, fontweight="bold", color=colors.get(id_str, "black"))

    # Draw origin axes (thicker)
    origin_len = cfg["marker_size_m"] * 1.2
    for axis_dir, color, label in [
        ([1, 0, 0], "red", "X"),
        ([0, 1, 0], "green", "Y"),
        ([0, 0, 1], "blue", "Z"),
    ]:
        d = np.array(axis_dir) * origin_len
        ax.quiver(0, 0, 0, *d, color=color, arrow_length_ratio=0.1, linewidth=3, alpha=0.4)
        ax.text(*(d * 1.15), label, color=color, fontsize=10, alpha=0.5)

    # Formatting
    all_pts = []
    for marker in cfg["markers"].values():
        corners, _, _ = marker_corners_3d(marker["position"], marker["quaternion_wxyz"], half)
        all_pts.extend(corners.tolist())
    all_pts = np.array(all_pts)
    center = all_pts.mean(axis=0)
    span = max(all_pts.max(axis=0) - all_pts.min(axis=0)) * 0.7
    ax.set_xlim(center[0] - span, center[0] + span)
    ax.set_ylim(center[1] - span, center[1] + span)
    ax.set_zlim(center[2] - span, center[2] + span)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Marker Config — ArUco 1 = Origin")
    ax.set_aspect("equal")

    out = config_path.rsplit(".", 1)[0] + "_viz.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved to {out}")
    plt.show()


if __name__ == "__main__":
    main()
