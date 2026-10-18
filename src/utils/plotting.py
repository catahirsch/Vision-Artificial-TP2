"""Plotting and visualization utilities for the stereo reconstruction notebooks.

Display rectification comparisons, disparity maps and Open3D scenes. Functions
with a ``save_path`` argument create the output directory when needed.
"""

from __future__ import annotations

import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import open3d as o3d

from .utils import get_image_at_index, imread_unicode
from .prints import print_pose_diagnostics
from ..calibration_rectification import rectify_image_pair
from ..reconstruction import estimate_global_pose, get_pair_index


def _save_figure(
    figure,
    save_path: str | None,
    *,
    dpi: int = 120,
    bbox_inches: str | None = "tight",
) -> None:
    """Save a Matplotlib figure, creating its parent directory when needed.

    Args:
        figure: Matplotlib figure to save.
        save_path: Output image path, or ``None`` to skip saving.
        dpi: Resolution of the saved image.
        bbox_inches: Bounding-box adjustment; ``None`` preserves the canvas size.
    """
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)


def show_rectified_pair(
    img_left: np.ndarray,
    img_right: np.ndarray,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    image_size: tuple[int, int],
    num_lines: int = 10,
    save_path: str | None = None,
):
    """Plot the original and rectified stereo pairs in a 2 x 2 grid.

    Draw horizontal reference lines on the rectified images to inspect epipolar
    alignment. Corresponding physical points should lie on the same image row.
    The figure is created and optionally saved; the caller handles display.

    Args:
        img_left: Original left BGR image.
        img_right: Original right BGR image.
        rect_left: Rectified left BGR image.
        rect_right: Rectified right BGR image.
        image_size: Image size as ``(width, height)``, in pixels.
        num_lines: Number of candidate line positions; the endpoints are omitted.
        save_path: Optional output image path.
    """
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(cv.cvtColor(img_left, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original left")
    axes[0, 1].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
    axes[0, 1].set_title("Original right")
    axes[1, 0].imshow(cv.cvtColor(rect_left, cv.COLOR_BGR2RGB))
    axes[1, 0].set_title("Rectified left")
    axes[1, 1].imshow(cv.cvtColor(rect_right, cv.COLOR_BGR2RGB))
    axes[1, 1].set_title("Rectified right")

    # Si el mismo punto cae en filas distintas, la rectificación todavía no cierra.
    for y in np.linspace(0, image_size[1], num_lines)[1:-1]:
        axes[1, 0].axhline(y, color="r", linestyle="--", alpha=0.6)
        axes[1, 1].axhline(y, color="r", linestyle="--", alpha=0.6)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    _save_figure(figure, save_path)


def show_disparity(
    disparity: np.ndarray,
    title: str = "Disparity map",
    cmap: str = "magma",
    save_path: str | None = None,
):
    """Display a disparity map with a color bar and optionally save the figure.

    Args:
        disparity: Floating-point disparity map, in pixels.
        title: Figure title.
        cmap: Matplotlib color map name.
        save_path: Optional output image path.
    """
    figure = plt.figure(figsize=(10, 5))
    plt.imshow(disparity, cmap=cmap)
    plt.title(title)
    plt.colorbar(label="disparity (px)")
    plt.axis("off")
    _save_figure(figure, save_path)
    plt.show()


def show_geometries(geometries: list, title: str = "Open3D"):
    """Open an interactive Open3D window for the supplied geometries.

    Args:
        geometries: Open3D point clouds, meshes or coordinate frames to display.
        title: Window title.
    """
    o3d.visualization.draw_geometries(geometries, window_name=title)


def show_camera_poses_top_view(
    camera_to_world_poses: np.ndarray,
    pair_ids: list[int],
    board,
    point_cloud: o3d.geometry.PointCloud | None = None,
    save_path: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
):
    """Plot camera centers and optical axes projected onto the world's XY plane.

    Args:
        camera_to_world_poses: Camera-to-world transforms with shape ``(N, 4, 4)``.
        pair_ids: Capture IDs in the same order as the transforms.
        board: ChArUco board defining the world frame and its physical dimensions.
        point_cloud: Optional world-space cloud, such as the optimized object.
        save_path: Optional output image path; saved at 220 dpi.
        xlim: Optional horizontal world-coordinate limits in millimeters.
        ylim: Optional vertical world-coordinate limits in millimeters.

    Returns:
        Matplotlib figure, also displayed for inline use in a notebook.
    """
    poses = np.asarray(camera_to_world_poses)
    if poses.shape != (len(pair_ids), 4, 4) or not len(pair_ids):
        raise ValueError("Provide at least one 4x4 pose and one ID per pose.")
    centers = poses[:, :3, 3]
    squares_x, squares_y = board.getChessboardSize()
    square_mm = board.getSquareLength()

    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False}):
        figure, axis = plt.subplots(figsize=(4.5, 5.0), layout="constrained")
        axis.add_patch(Rectangle(
            (0, 0), squares_x * square_mm, squares_y * square_mm,
            fill=False, linestyle="--", linewidth=1.2, edgecolor="0.45",
            label="Tablero ChArUco"
        ))
        if point_cloud is not None and len(point_cloud.points):
            points = np.asarray(point_cloud.points)[::12]
            colors = np.asarray(point_cloud.colors)[::12] if point_cloud.has_colors() else "0.5"
            axis.scatter(points[:, 0], points[:, 1], c=colors, s=1,
                         rasterized=True, zorder=2)
        axis.plot(centers[:, 0], centers[:, 1], color="0.7", linewidth=0.8,
                  zorder=1)
        cameras = axis.scatter(centers[:, 0], centers[:, 1], c=pair_ids,
                               cmap="viridis", s=24, zorder=3)
        # The third rotation column is the camera's +Z optical axis in world space.
        sample = np.arange(0, len(centers), 7)
        optical = poses[sample, :2, 2]
        axis.quiver(centers[sample, 0], centers[sample, 1], optical[:, 0],
                    optical[:, 1], angles="xy", scale_units="xy", scale=1 / 70,
                    color="0.25", width=0.005, zorder=3)
        for index, offset in [(0, (8, 7)), (-1, (8, -15))]:
            axis.annotate(f"Par {pair_ids[index]}", centers[index, :2],
                          xytext=offset, textcoords="offset points", fontsize=9)
        axis.scatter([0], [0], color="black", marker="+", s=60, zorder=4)
        axis.annotate("Origen", (0, 0), xytext=(7, -15),
                      textcoords="offset points", fontsize=9)
        axis.set(xlabel="X del mundo (mm)", ylabel="Y del mundo (mm)",
                 xlim=xlim, ylim=ylim)
        axis.set_aspect("equal")
        axis.grid(alpha=0.2)
        axis.legend(loc="lower left", bbox_to_anchor=(0, 1.02), fontsize=9,
                    framealpha=0.9)
        colorbar = figure.colorbar(cameras, ax=axis, shrink=0.75, pad=0.03)
        colorbar.set_label("Índice de captura")
        _save_figure(figure, save_path, dpi=220, bbox_inches=None)
        plt.show()
    return figure


def show_global_pose_example(
    images_left: list[str],
    maps: dict,
    board,
    detector,
    point_cloud: o3d.geometry.PointCloud | None = None,
    save_path: str | None = None,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> dict:
    """Estimate capture poses and display the report's global-frame figure.

    Use the rectified grayscale images and pose solver used by the global merger.
    Print pose counts and reprojection diagnostics from the detected ChArUco corners.
    This only estimates poses and draws the supplied cloud; it does not rerun fusion.

    Args:
        images_left: Original left image paths, sorted by capture ID internally.
        maps: Saved rectification maps, including the projection matrix ``P1``.
        board: ChArUco world-reference board.
        detector: Detector associated with ``board``.
        point_cloud: Optional world-space cloud to overlay on the camera positions.
        save_path: Optional output image path for the report.
        xlim: Optional X world-coordinate limits in millimeters.
        ylim: Optional Y world-coordinate limits in millimeters.

    Returns:
        Accepted IDs and poses, rejected IDs, and per-capture reprojection RMS in pixels.
    """
    intrinsic = np.asarray(maps["P1"])[:3, :3]
    pair_ids, poses, rejected, errors = [], [], [], []
    for path in sorted(images_left, key=get_pair_index):
        pair_id = get_pair_index(path)
        image = imread_unicode(path, cv.IMREAD_GRAYSCALE)
        if image is None:
            raise OSError(f"Could not load image: {path}")
        rectified = cv.remap(
            image, maps["left_map_x"], maps["left_map_y"], cv.INTER_LINEAR
        )
        pose = estimate_global_pose(rectified, detector, board, intrinsic)
        if pose is None:
            rejected.append(pair_id)
            continue
        pair_ids.append(pair_id)
        poses.append(pose)
        corners, ids, _, _ = detector.detectBoard(rectified)
        object_points, image_points = board.matchImagePoints(corners, ids)
        world_to_camera = np.linalg.inv(pose)
        rotation, _ = cv.Rodrigues(world_to_camera[:3, :3])
        projected, _ = cv.projectPoints(
            object_points, rotation, world_to_camera[:3, 3], intrinsic, np.zeros(5)
        )
        difference = projected.reshape(-1, 2) - image_points.reshape(-1, 2)
        errors.append(np.sqrt(np.mean(np.sum(difference**2, axis=1))))

    poses = np.asarray(poses)
    show_camera_poses_top_view(
        poses, pair_ids, board, point_cloud=point_cloud, save_path=save_path,
        xlim=xlim, ylim=ylim,
    )
    diagnostics = {
        "pair_ids": pair_ids,
        "poses": poses,
        "rejected_pair_ids": rejected,
        "reprojection_rms_px": np.asarray(errors),
    }
    print_pose_diagnostics(diagnostics)
    return diagnostics


def save_open3d_screenshot(
    geometries: list,
    save_path: str,
    width: int = 1280,
    height: int = 960,
) -> None:
    """Render an Open3D scene to an image without opening a visible window.

    Use this to generate report figures such as the merged cloud and camera poses.

    Args:
        geometries: Open3D geometries to render.
        save_path: Output image path, usually with a ``.png`` extension.
        width: Screenshot width, in pixels.
        height: Screenshot height, in pixels.
    """
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=False, width=width, height=height)
    for g in geometries:
        visualizer.add_geometry(g)
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(save_path, do_render=True)
    visualizer.destroy_window()


def show_calibration_errors(diagnostics: dict, save_path: str | None = None):
    """Plot reprojection and rectified vertical errors for each capture.

    Args:
        diagnostics: Calibration diagnostics containing per-capture metrics in ``rows``.
        save_path: Optional output image path; the figure is also displayed.
    """
    rows = diagnostics["rows"]
    used_pair_ids = [row["pair_id"] for row in rows]
    per_view_rms = np.array(
        [
            [row["left_reprojection_rms_px"], row["right_reprojection_rms_px"]]
            for row in rows
        ]
    )
    diagnostics = rows
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(used_pair_ids, per_view_rms[:, 0], ".-", label="Left")
    axes[0].plot(
        used_pair_ids, per_view_rms[:, 1], ".-", label="Right from left-camera pose"
    )
    axes[0].set_ylabel("Reprojection RMS (px)")
    axes[0].legend()
    axes[1].plot(
        used_pair_ids, [row["vertical_alignment_rms_px"] for row in diagnostics], ".-"
    )
    axes[1].set_ylabel("Vertical RMS (px)")
    axes[1].set_xlabel("Calibration pair index")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    _save_figure(figure, save_path)
    plt.show()


def show_rectification_example(
    dataset_name: str,
    index: int,
    maps: dict,
    image_size: tuple[int, int],
    save_path: str | None = None,
):
    """Load a dataset pair, rectify it and display the comparison figure.

    Args:
        dataset_name: Dataset directory name under ``data/raw``.
        index: Stereo capture number to load.
        maps: Stereo rectification maps returned by ``rectify``.
        image_size: Calibration image size as ``(width, height)``.
        save_path: Optional output image path.

    Raises:
        ValueError: If the example image size differs from the calibration size.
    """
    img_left, img_right = get_image_at_index(dataset_name=dataset_name, index=index)
    if img_left.shape[1::-1] != image_size or img_right.shape[1::-1] != image_size:
        raise ValueError(
            "The example image size differs from the calibration image size."
        )
    rect_left, rect_right = rectify_image_pair(img_left, img_right, maps)
    figure = show_rectified_pair(
        img_left,
        img_right,
        rect_left,
        rect_right,
        image_size,
        save_path=save_path,
    )
    plt.show()
