"""Stereo calibration, rectification and reprojection diagnostics for notebook 01."""

from collections.abc import Mapping, Sequence
from pathlib import Path
import csv
import hashlib
import pickle
from datetime import datetime, timezone

import cv2
import numpy as np
import yaml

from .utils.utils import imread_unicode, indexed_images


def get_image_points(
    square_size: float,
    rows: int,
    cols: int,
    images_right: Sequence[str | Path],
    images_left: Sequence[str | Path],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[int]]:
    """
    Detect and refine checkerboard corners in pairs seen by both cameras.

    Args:
        square_size: Checkerboard square side length in millimeters.
        rows: Number of inner checkerboard corners per column.
        cols: Number of inner checkerboard corners per row.
        images_right: Right image paths, ordered to match ``images_left``.
        images_left: Left image paths, ordered to match ``images_right``.

    Returns:
        Object points and the corresponding detected image points for the left
        and right cameras, and indices of accepted pairs in the input lists.
        A pair is included only when both images contain a detected checkerboard.
    """
    if not images_left or len(images_left) != len(images_right):
        raise ValueError("Provide non-empty left/right lists with matching lengths.")
    if square_size <= 0 or rows < 2 or cols < 2:
        raise ValueError("The square size must be positive and the grid at least 2x2.")
    # Cada cuadrado fija la escala: si va en mm, la reconstrucción también.
    object_points_for_pair = np.zeros((rows * cols, 3), np.float32)
    object_points_for_pair[:, :2] = (
        np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points, image_points_left, image_points_right, accepted_indices = (
        [],
        [],
        [],
        [],
    )
    expected_shape = None
    for index, (img_left_path, img_right_path) in enumerate(
        zip(images_left, images_right)
    ):
        img_left = imread_unicode(img_left_path, cv2.IMREAD_GRAYSCALE)
        img_right = imread_unicode(img_right_path, cv2.IMREAD_GRAYSCALE)
        if img_left is None or img_right is None:
            raise OSError(
                f"Could not read stereo pair: {img_left_path}, {img_right_path}"
            )
        if expected_shape is None:
            expected_shape = img_left.shape
        if img_left.shape != expected_shape or img_right.shape != expected_shape:
            raise ValueError(
                f"Image size differs in pair: {img_left_path}, {img_right_path}"
            )
        found_left, corners_left = cv2.findChessboardCorners(
            img_left, (cols, rows), None
        )
        found_right, corners_right = cv2.findChessboardCorners(
            img_right, (cols, rows), None
        )
        if found_left and found_right:
            accepted_indices.append(index)
            object_points.append(object_points_for_pair)
            image_points_left.append(
                cv2.cornerSubPix(img_left, corners_left, (11, 11), (-1, -1), criteria)
            )
            image_points_right.append(
                cv2.cornerSubPix(img_right, corners_right, (11, 11), (-1, -1), criteria)
            )

    return object_points, image_points_left, image_points_right, accepted_indices


def calibrate(
    object_points: Sequence[np.ndarray],
    image_points_left: Sequence[np.ndarray],
    image_points_right: Sequence[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[dict[str, np.ndarray | tuple[int, int]], dict[str, float]]:
    """
    Calibrate both cameras, then estimate their relative rotation and translation.

    Intrinsics and distortion are kept fixed during stereo calibration, so the
    monocular RMS values describe the same intrinsics returned to the notebook.

    Args:
        object_points: Checkerboard 3D points for each accepted image pair.
        image_points_left: Detected checkerboard corners in left images.
        image_points_right: Detected checkerboard corners in right images.
        image_size: Image size as ``(width, height)``.

    Returns:
        Calibration in the stereo_calibration.pkl schema, and the left, right
        and stereo calibration RMS values in pixels.
    """
    if len(object_points) < 3:
        raise ValueError(
            "At least three detected stereo views are required to calibrate."
        )
    if not (len(object_points) == len(image_points_left) == len(image_points_right)):
        raise ValueError("Object and image point lists must have matching lengths.")

    rms_left, K1, D1, _, _ = cv2.calibrateCamera(
        object_points, image_points_left, image_size, None, None
    )
    rms_right, K2, D2, _, _ = cv2.calibrateCamera(
        object_points, image_points_right, image_size, None, None
    )
    rms_stereo, K1, D1, K2, D2, R, t, E, F = cv2.stereoCalibrate(
        object_points,
        image_points_left,
        image_points_right,
        K1,
        D1,
        K2,
        D2,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )
    calibration = {
        "left_K": K1,
        "right_K": K2,
        "left_dist": D1,
        "right_dist": D2,
        "R": R,
        "T": t,
        "E": E,
        "F": F,
        "image_size": image_size,
    }
    errors = {
        "left": float(rms_left),
        "right": float(rms_right),
        "stereo": float(rms_stereo),
    }
    if not all(np.isfinite(value).all() for value in calibration.values()):
        raise RuntimeError("Calibration returned non-finite parameters.")
    return calibration, errors


def stereo_reprojection_errors(
    object_points: Sequence[np.ndarray],
    image_points_left: Sequence[np.ndarray],
    image_points_right: Sequence[np.ndarray],
    K1: np.ndarray,
    D1: np.ndarray,
    K2: np.ndarray,
    D2: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> np.ndarray:
    """
    Validate fixed parameters and return per-view RMS for each camera (N x 2).

    Estimate the checkerboard pose in the left camera, transform it to the
    right camera using ``R`` and ``t``, then compare projected points with the
    detected corners in both images.

    Args:
        object_points: Checkerboard 3D points for each accepted image pair.
        image_points_left: Detected checkerboard corners in left images.
        image_points_right: Detected checkerboard corners in right images.
        K1: Left camera intrinsic matrix.
        D1: Left camera distortion coefficients.
        K2: Right camera intrinsic matrix.
        D2: Right camera distortion coefficients.
        R: Rotation from the left camera coordinate system to the right.
        t: Translation from the left camera coordinate system to the right.

    Returns:
        Left and right RMS errors per view. The board pose is fitted to the left
        image only; this diagnostic differs from stereoCalibrate's joint RMS.
    """
    if not object_points or not (
        len(object_points) == len(image_points_left) == len(image_points_right)
    ):
        raise ValueError("Provide non-empty object/image point lists of equal length.")
    errors = []
    t = np.asarray(t).reshape(3, 1)
    for object_points_for_pair, points_left, points_right in zip(
        object_points, image_points_left, image_points_right
    ):
        # Ajustamos la pose solo con la izquierda; la derecha sirve de control.
        ok, rvec, tvec = cv2.solvePnP(object_points_for_pair, points_left, K1, D1)
        if not ok:
            raise RuntimeError("Could not estimate a checkerboard pose for validation.")
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
        errors.append(
            [
                np.sqrt(np.mean(np.sum(diff_left**2, axis=1))),
                np.sqrt(np.mean(np.sum(diff_right**2, axis=1))),
            ]
        )

    return np.asarray(errors)


def rectify(
    K1: np.ndarray,
    K2: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    image_size: tuple[int, int],
    R: np.ndarray,
    t: np.ndarray,
    alpha: float = 0.0,
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
        alpha: Rectification scaling. Zero crops invalid borders; one keeps all pixels.

    Returns:
        A dictionary matching the ``stereo_maps.pkl`` format: ``left_map_x``,
        ``left_map_y``, ``right_map_x`` and ``right_map_y`` are float32 maps
        for ``cv2.remap``; ``R1`` and ``R2`` are rectification rotations;
        ``P1`` and ``P2`` are rectified projection matrices; ``Q`` maps
        disparity to 3D; ``validRoi1`` and ``validRoi2`` are valid image regions
        as ``(x, y, width, height)``.
    """
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size, R, t, alpha=alpha
    )
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


def load_calibration_data(
    images_dir: str | Path, square_size: float, rows: int, cols: int
) -> dict:
    """Pair calibration images by capture number and detect checkerboard corners.

    Args:
        images_dir: Directory containing ``left_<n>.jpg`` and ``right_<n>.jpg`` pairs.
        square_size: Checkerboard square side length, in millimeters.
        rows: Number of inner checkerboard corners per column.
        cols: Number of inner checkerboard corners per row.

    Returns:
        Calibration inputs containing image paths, image size, object and image
        points, and the available, accepted and rejected capture IDs.

    Raises:
        FileNotFoundError: If no left calibration images are found.
        ValueError: If left and right capture IDs do not match.
    """
    images_dir = Path(images_dir)
    left_by_index = indexed_images(images_dir, "left")
    right_by_index = indexed_images(images_dir, "right")
    if not left_by_index:
        raise FileNotFoundError(f"No images found in {images_dir}")
    if left_by_index.keys() != right_by_index.keys():
        raise ValueError(
            f"Incomplete pairs. Missing right images: {sorted(left_by_index.keys() - right_by_index.keys())}; "
            f"missing left images: {sorted(right_by_index.keys() - left_by_index.keys())}"
        )

    pair_ids = sorted(left_by_index)
    images_left = [left_by_index[index] for index in pair_ids]
    images_right = [right_by_index[index] for index in pair_ids]
    object_points, image_points_left, image_points_right, accepted_indices = (
        get_image_points(
            square_size=square_size,
            rows=rows,
            cols=cols,
            images_left=images_left,
            images_right=images_right,
        )
    )
    used_pair_ids = [pair_ids[index] for index in accepted_indices]
    used_ids_set = set(used_pair_ids)
    rejected_pair_ids = [index for index in pair_ids if index not in used_ids_set]
    image_size = imread_unicode(images_left[0], cv2.IMREAD_GRAYSCALE).shape[::-1]

    return {
        "directory": images_dir,
        "pair_ids": pair_ids,
        "images_left": images_left,
        "images_right": images_right,
        "object_points": object_points,
        "image_points_left": image_points_left,
        "image_points_right": image_points_right,
        "image_size": image_size,
        "used_pair_ids": used_pair_ids,
        "rejected_pair_ids": rejected_pair_ids,
    }


def evaluate_calibration(
    calib: Mapping, maps: Mapping, calibration_data: Mapping
) -> dict:
    """Compute reprojection and rectified vertical errors for each accepted capture.

    Args:
        calib: Stereo calibration matrices and distortion coefficients.
        maps: Rectification rotations and projection matrices returned by ``rectify``.
        calibration_data: Accepted object points, image points and capture IDs.

    Returns:
        Per-capture metrics in ``rows``, point-weighted stereo ``validation_rms``
        and aggregate rectified ``vertical_rms``, all in pixels.
    """
    K1, D1 = calib["left_K"], calib["left_dist"]
    K2, D2 = calib["right_K"], calib["right_dist"]
    R, t = calib["R"], calib["T"]
    object_points = calibration_data["object_points"]
    image_points_left = calibration_data["image_points_left"]
    image_points_right = calibration_data["image_points_right"]
    used_pair_ids = calibration_data["used_pair_ids"]
    per_view_rms = stereo_reprojection_errors(
        object_points, image_points_left, image_points_right, K1, D1, K2, D2, R, t
    )
    vertical_errors = []
    diagnostics = []
    for index, points_left, points_right, errors in zip(
        used_pair_ids, image_points_left, image_points_right, per_view_rms
    ):
        rectified_points_left = cv2.undistortPoints(
            points_left, K1, D1, R=maps["R1"], P=maps["P1"]
        ).reshape(-1, 2)
        rectified_points_right = cv2.undistortPoints(
            points_right, K2, D2, R=maps["R2"], P=maps["P2"]
        ).reshape(-1, 2)
        dy = rectified_points_left[:, 1] - rectified_points_right[:, 1]
        vertical_errors.append(dy)
        diagnostics.append(
            {
                "pair_id": index,
                "left_reprojection_rms_px": float(errors[0]),
                "right_reprojection_rms_px": float(errors[1]),
                "vertical_alignment_rms_px": float(np.sqrt(np.mean(dy**2))),
            }
        )

    point_counts = np.array([len(points) for points in object_points])
    validation_rms = float(
        np.sqrt(np.average(np.mean(per_view_rms**2, axis=1), weights=point_counts))
    )
    all_vertical_errors = np.concatenate(vertical_errors)
    vertical_rms = float(np.sqrt(np.mean(all_vertical_errors**2)))
    return {
        "rows": diagnostics,
        "validation_rms": validation_rms,
        "vertical_rms": vertical_rms,
    }


def save_calibration_results(
    output_dir: str | Path,
    calib: Mapping,
    maps: Mapping,
    calibration_rms: Mapping,
    calibration_data: Mapping,
    diagnostics: Mapping,
    *,
    root_dir: str | Path,
    rows: int,
    cols: int,
    square_size_mm: float,
    rectification_alpha: float,
) -> dict:
    """Save calibration matrices, rectification maps, diagnostics and provenance.

    Args:
        output_dir: Directory for the pickle, CSV and YAML artifacts.
        calib: Stereo calibration parameters to save.
        maps: Stereo rectification maps and matrices to save.
        calibration_rms: Left, right and stereo calibration RMS values, in pixels.
        calibration_data: Image directory, capture IDs and accepted calibration data.
        diagnostics: Aggregate validation errors and per-capture metrics in ``rows``.
        root_dir: Project root used to store the relative calibration image directory.
        rows: Number of inner checkerboard corners per column.
        cols: Number of inner checkerboard corners per row.
        square_size_mm: Checkerboard square side length, in millimeters.
        rectification_alpha: Rectification scaling used to generate the maps.

    Returns:
        Metadata written to ``metadata.yaml``, including artifact sizes and hashes.
        The pickle artifacts follow the schema consumed by notebook 02.
    """
    output_dir = Path(output_dir)
    root_dir = Path(root_dir)
    images_dir = calibration_data["directory"]
    pair_ids = calibration_data["pair_ids"]
    used_pair_ids = calibration_data["used_pair_ids"]
    rejected_pair_ids = calibration_data["rejected_pair_ids"]
    image_size = calib["image_size"]
    t = calib["T"]
    validation_rms = diagnostics["validation_rms"]
    vertical_rms = diagnostics["vertical_rms"]
    diagnostics = diagnostics["rows"]
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, results in [
        ("stereo_calibration.pkl", calib),
        ("stereo_maps.pkl", maps),
    ]:
        with (output_dir / filename).open("wb") as stream:
            pickle.dump(results, stream, protocol=pickle.HIGHEST_PROTOCOL)

    with (output_dir / "reprojection_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows(diagnostics)

    metadata = {
        "schema_version": 1,
        "status": "Calibration generated in notebook 01; review quality and validate physical scale.",
        "provenance": {
            "tool": "OpenCV via notebooks/01_calibration.ipynb",
            "calibration_date": datetime.now(timezone.utc).isoformat(),
            "generation_dependency_versions": {
                "opencv": cv2.__version__,
                "numpy": np.__version__,
            },
            "methods": [
                "calibrateCamera (each camera)",
                "stereoCalibrate (CALIB_FIX_INTRINSIC)",
            ],
        },
        "image_size_per_camera_px": {
            "width": int(image_size[0]),
            "height": int(image_size[1]),
        },
        "calibration_images": {
            "directory_from_project_root": str(images_dir.relative_to(root_dir)),
            "available_stereo_pairs": len(pair_ids),
            "pairs_used_for_saved_calibration": len(used_pair_ids),
            "used_pair_ids": used_pair_ids,
            "rejected_pair_ids": rejected_pair_ids,
            "correspondence_with_saved_results_confirmed": True,
        },
        "checkerboard": {
            "inner_corners_columns": cols,
            "inner_corners_rows": rows,
            "squares_columns": cols + 1,
            "squares_rows": rows + 1,
            "square_size_mm": square_size_mm,
            "pattern_width_mm": (cols + 1) * square_size_mm,
            "pattern_height_mm": (rows + 1) * square_size_mm,
            "source": "Configuration in notebooks/01_calibration.ipynb",
        },
        "quality": {
            "left_monocular_calibration_rms_px": calibration_rms["left"],
            "right_monocular_calibration_rms_px": calibration_rms["right"],
            "stereo_reprojection_rms_px": calibration_rms["stereo"],
            "stereo_validation_left_pose_rms_px": validation_rms,
            "rectified_vertical_alignment_rms_px": vertical_rms,
        },
        "metric_scale": {
            "length_unit": "mm based on square_size_mm",
            "physical_square_size_confirmed": True,
            "baseline_in_calibration_units": float(np.linalg.norm(t)),
            "note": "Confirm the printed board square size is 28 mm and validate a known distance before measuring the object.",
        },
        "rectification": {"alpha": rectification_alpha},
        "artifacts": {},
    }
    for filename in [
        "stereo_calibration.pkl",
        "stereo_maps.pkl",
        "reprojection_errors.csv",
    ]:
        artifact = output_dir / filename
        metadata["artifacts"][filename] = {
            "bytes": artifact.stat().st_size,
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        }
    with (output_dir / "metadata.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, allow_unicode=True, sort_keys=False)

    return metadata


def calibrate_and_evaluate(
    calibration_data: Mapping,
    image_size: tuple[int, int],
    alpha: float,
    tau: float,
    max_iter: int,
) -> dict:
    """Calibrate repeatedly, rejecting stereo views with reprojection RMS > tau.

    ``tau`` is a positive threshold in pixels, applied to each camera's per-view
    reprojection RMS. A pair is retained only if both errors are finite and at
    most ``tau``. Vertical alignment is reported but does not control rejection.
    ``max_iter`` limits the number of calibrations, including the first one.

    Args:
        calibration_data: Object/image points, capture IDs and source image paths.
        image_size: Calibration image size as ``(width, height)``.
        alpha: Rectification scaling; zero crops invalid borders.
        tau: Maximum accepted per-camera, per-view reprojection RMS, in pixels.
        max_iter: Maximum number of calibration iterations, including the first.

    Returns:
        A dictionary with ``calib``, ``calibration_rms``, ``maps``, ``diagnostics``
        and ``filtered_data``. Stop when every retained pair meets the threshold,
        or the iteration budget is exhausted, returning the last evaluated
        calibration.
        The returned data and diagnostics always describe that same calibration.
        Input data is not modified; original image paths and available pair IDs
        remain in ``filtered_data``, with updated used/rejected IDs and points.
    """
    point_keys = ("object_points", "image_points_left", "image_points_right")

    filtered_data = dict(calibration_data)
    for key in (*point_keys, "used_pair_ids", "rejected_pair_ids"):
        filtered_data[key] = list(calibration_data[key])

    ready = False
    for i in range(max_iter):
        calib, calibration_rms = calibrate(
            filtered_data["object_points"],
            filtered_data["image_points_left"],
            filtered_data["image_points_right"],
            image_size,
        )

        maps = rectify(
            K1=calib["left_K"],
            K2=calib["right_K"],
            D1=calib["left_dist"],
            D2=calib["right_dist"],
            image_size=image_size,
            R=calib["R"],
            t=calib["T"],
            alpha=alpha,
        )
        diagnostics = evaluate_calibration(calib, maps, filtered_data)
        # Ojo: el par se queda solo si las dos cámaras pasan el umbral.
        keep_indices = [
            index
            for index, row in enumerate(diagnostics["rows"])
            if all(
                row[key] <= tau
                for key in (
                    "left_reprojection_rms_px",
                    "right_reprojection_rms_px",
                )
            )
        ]
        ready = len(keep_indices) == len(filtered_data["used_pair_ids"])

        if ready or i == max_iter - 1:
            break

        filtered_data["rejected_pair_ids"] = sorted(
            filtered_data["rejected_pair_ids"]
            + [
                pair_id
                for index, pair_id in enumerate(filtered_data["used_pair_ids"])
                if index not in keep_indices
            ]
        )
        for key in (*point_keys, "used_pair_ids"):
            filtered_data[key] = [filtered_data[key][index] for index in keep_indices]

    return {
        "calib": calib,
        "calibration_rms": calibration_rms,
        "maps": maps,
        "diagnostics": diagnostics,
        "filtered_data": filtered_data,
    }
