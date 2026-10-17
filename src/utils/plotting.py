"""Plotting and visualization utilities for the stereo reconstruction notebooks.

Display rectification comparisons, disparity maps and Open3D scenes. Functions
with a ``save_path`` argument create the output directory when needed.
"""

from __future__ import annotations

import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d

from .utils import get_image_at_index
from ..calibration_rectification import rectify_image_pair


def _save_figure(figure, save_path: str | None) -> None:
    """Save a Matplotlib figure, creating its parent directory when needed.

    Args:
        figure: Matplotlib figure to save.
        save_path: Output image path, or ``None`` to skip saving.
    """
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(save_path, dpi=120, bbox_inches="tight")


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
