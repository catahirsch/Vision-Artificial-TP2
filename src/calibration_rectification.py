"""Helpers for detecting calibration points and measuring reprojection error."""

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np


def get_image_points(
    square_size: float,
    rows: int,
    cols: int,
    images_right: Sequence[str | Path],
    images_left: Sequence[str | Path],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Detect and refine checkerboard corners in pairs seen by both cameras.

    Args:
        sq_size: Checkerboard square side length in millimeters.
        rows: Number of inner checkerboard corners per column.
        cols: Number of inner checkerboard corners per row.
        images_right: Right image paths, ordered to match ``images_left``.
        images_left: Left image paths, ordered to match ``images_right``.

    Returns:
        Object points and the corresponding detected image points for the left
        and right cameras, respectively. A pair is included only when both
        images contain a detected checkerboard.
    """
    object_points_for_pair = np.zeros((rows * cols, 3), np.float32)
    object_points_for_pair[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points, image_points_left, image_points_right = [], [], []
    for img_left_path, img_right_path in zip(images_left, images_right):
        img_left = cv2.imread(str(img_left_path), cv2.IMREAD_GRAYSCALE)
        img_right = cv2.imread(str(img_right_path), cv2.IMREAD_GRAYSCALE)
        found_left, corners_left = cv2.findChessboardCorners(
            img_left, (cols, rows), None
        )
        found_right, corners_right = cv2.findChessboardCorners(
            img_right, (cols, rows), None
        )
        if found_left and found_right:
            object_points.append(object_points_for_pair)
            image_points_left.append(
                cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1), criteria)
            )
            image_points_right.append(
                cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)
            )

    return object_points, image_points_left, image_points_right


def calibrate(
    object_points: Sequence[np.ndarray],
    image_points_left: Sequence[np.ndarray],
    image_points_right: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """Calibrate each camera independently and return its RMS error in pixels.

    Args:
        objpoints: Checkerboard 3D points for each accepted image pair.
        imgpoints_left: Detected checkerboard corners in left images.
        imgpoints_right: Detected checkerboard corners in right images.
        image_size: Image size as ``(width, height)``.

    Returns:
        The monocular calibration RMS values for the left and right cameras.
    """
    rms_left, *_ = cv2.calibrateCamera(
        object_points, image_points_left, image_size, None, None
    )
    rms_right, *_ = cv2.calibrateCamera(
        object_points, image_points_right, image_size, None, None
    )
    return rms_left, rms_right


def rms_reprojection_error(
    object_points: Sequence[np.ndarray],
    image_points_left: Sequence[np.ndarray],
    image_points_right: Sequence[np.ndarray],
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> float:
    """Compute stereo reprojection RMS in pixels using saved calibration data.

    Estimate the checkerboard pose in the left camera, transform it to the
    right camera using ``R`` and ``t``, then compare projected points with the
    detected corners in both images.

    Args:
        objpoints: Checkerboard 3D points for each accepted image pair.
        imgpoints_left: Detected checkerboard corners in left images.
        imgpoints_right: Detected checkerboard corners in right images.
        K1: Left camera intrinsic matrix.
        D1: Left camera distortion coefficients.
        K2: Right camera intrinsic matrix.
        D2: Right camera distortion coefficients.
        R: Rotation from the left camera coordinate system to the right.
        t: Translation from the left camera coordinate system to the right.

    Returns:
        Root mean square reprojection error in pixels across both cameras.
    """
    squared_error, point_count = 0.0, 0
    for object_points_for_pair, points_left, points_right in zip(
        object_points, image_points_left, image_points_right
    ):
        _, rvec, tvec = cv2.solvePnP(object_points_for_pair, points_left, K1, D1)
        projected_points_left, _ = cv2.projectPoints(
            object_points_for_pair, rvec, tvec, K1, D1
        )
        board_rotation, _ = cv2.Rodrigues(rvec)
        rvec_right, _ = cv2.Rodrigues(R @ board_rotation)
        tvec_right = R @ tvec + t
        projected_points_right, _ = cv2.projectPoints(
            object_points_for_pair, rvec_right, tvec_right, K2, D2
        )
        diff_left = projected_points_left.reshape(-1, 2) - points_left.reshape(-1, 2)
        diff_right = projected_points_right.reshape(-1, 2) - points_right.reshape(-1, 2)
        squared_error += float((diff_left**2).sum() + (diff_right**2).sum())
        point_count += 2 * len(object_points_for_pair)

    return np.sqrt(squared_error / point_count)


def rectify(
    K1: np.ndarray,
    K2: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    image_size: tuple[int, int],
    R: np.ndarray,
    t: np.ndarray,
) -> dict[str, np.ndarray | tuple[int, int, int, int]]:
    """Compute stereo rectification matrices and image remapping coordinates.

    Args:
        K1: Left camera intrinsic matrix.
        K2: Right camera intrinsic matrix.
        D1: Left camera distortion coefficients.
        D2: Right camera distortion coefficients.
        image_size: Image size as ``(width, height)``.
        R: Rotation from the left camera coordinate system to the right.
        t: Translation from the left camera coordinate system to the right.

    Returns:
        A dictionary matching the ``stereo_maps.pkl`` format: ``left_map_x``,
        ``left_map_y``, ``right_map_x`` and ``right_map_y`` are float32 maps
        for ``cv2.remap``; ``R1`` and ``R2`` are rectification rotations;
        ``P1`` and ``P2`` are rectified projection matrices; ``Q`` maps
        disparity to 3D; ``validRoi1`` and ``validRoi2`` are valid image regions
        as ``(x, y, width, height)``.
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K1, D1, K2, D2, image_size, R, t)
    left_rectification_map_x, left_rectification_map_y = cv2.initUndistortRectifyMap(
        K1, D1, R1, P1, image_size, cv2.CV_32FC1
    )
    right_rectification_map_x, right_rectification_map_y = cv2.initUndistortRectifyMap(
        K2, D2, R2, P2, image_size, cv2.CV_32FC1
    )

    return {
        "left_map_x": left_rectification_map_x,
        "left_map_y": left_rectification_map_y,
        "right_map_x": right_rectification_map_x,
        "right_map_y": right_rectification_map_y,
        "R1": R1,
        "R2": R2,
        "P1": P1,
        "P2": P2,
        "Q": Q,
        "validRoi1": roi1,
        "validRoi2": roi2,
    }


def rectify_image_pair(
    img_left: np.ndarray,
    img_right: np.ndarray,
    rectification_maps: Mapping[str, np.ndarray | tuple[int, int, int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply stereo rectification maps to a left/right image pair.

    Args:
        img_left: Original left image.
        img_right: Original right image.
        rectification_maps: Maps returned by ``rectify`` or loaded from
            ``stereo_maps.pkl``. Requires ``left_map_x``, ``left_map_y``,
            ``right_map_x`` and ``right_map_y``.

    Returns:
        Left and right rectified images, remapped with bilinear interpolation.
    """
    rect_left = cv2.remap(
        img_left,
        rectification_maps["left_map_x"],
        rectification_maps["left_map_y"],
        cv2.INTER_LINEAR,
    )
    rect_right = cv2.remap(
        img_right,
        rectification_maps["right_map_x"],
        rectification_maps["right_map_y"],
        cv2.INTER_LINEAR,
    )
    return rect_left, rect_right
