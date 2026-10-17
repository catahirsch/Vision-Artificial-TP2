"""Stereo disparity, local point clouds and ChArUco-based global reconstruction.

Utilities for ``notebooks/02_reconstruction.ipynb`` receive rectification maps,
reprojection matrices and camera parameters explicitly, without notebook globals.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import cv2 as cv
import open3d as o3d

from .utils.utils import imread_unicode, imwrite_unicode


def get_pair_index(path: str) -> int:
    """Extract the capture number from a stereo image filename.

    Use this index to sort pairs numerically rather than lexicographically.

    Args:
        path: Image path or filename in the form ``left_<n>.jpg`` or ``right_<n>.jpg``.

    Returns:
        Capture number encoded in the filename.
    """
    return int(os.path.basename(path).split("_")[1].split(".")[0])


def build_local_point_cloud(
    left_path: str,
    disparity: np.ndarray,
    Q: np.ndarray,
    left_rectification_map_x: np.ndarray,
    left_rectification_map_y: np.ndarray,
    min_disparity: int = 0,
    z_min: float = 100.0,
    z_max: float = 2000.0,
) -> o3d.geometry.PointCloud:
    """Build a colored point cloud in the rectified left camera frame.

    Reproject pixel disparities with ``Q``, assign colors from the rectified left
    image and discard invalid matches, non-finite points and out-of-range depths.

    Args:
        left_path: Path to the original left image.
        disparity: Floating-point disparity map, in pixels; zero marks invalid pixels.
        Q: Rectification reprojection matrix with shape ``(4, 4)``.
        left_rectification_map_x: Horizontal remapping coordinates for the left camera.
        left_rectification_map_y: Vertical remapping coordinates for the left camera.
        min_disparity: Disparity threshold, in pixels; only larger values are retained.
        z_min: Exclusive lower depth bound, in millimeters.
        z_max: Exclusive upper depth bound, in millimeters.

    Returns:
        Colored point cloud in the rectified left camera coordinate system.
    """
    left_image = imread_unicode(left_path, cv.IMREAD_COLOR)
    rectified_left_color = cv.remap(
        left_image,
        left_rectification_map_x,
        left_rectification_map_y,
        cv.INTER_LINEAR,
    )

    points_3d = cv.reprojectImageTo3D(disparity, Q)  # (H, W, 3) en mm
    colors = (
        cv.cvtColor(rectified_left_color, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0
    )

    # Filtrado: disparidad válida, coordenadas finitas y profundidad en rango
    mask = (disparity > min_disparity) & np.isfinite(points_3d).all(axis=2)
    mask &= (points_3d[:, :, 2] > z_min) & (points_3d[:, :, 2] < z_max)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points_3d[mask])
    point_cloud.colors = o3d.utility.Vector3dVector(colors[mask])
    return point_cloud


def create_charuco_detector(
    squares_x: int = 9,
    squares_y: int = 13,
    square_mm: float = 42.0,
    marker_mm: float = 31.5,
    dict_id: int = cv.aruco.DICT_5X5_100,
):
    """Create the ChArUco world-reference board and its detector.

    Defaults match the reconstruction captures: a 9 x 13 board with 42 mm squares,
    31.5 mm markers and the ``DICT_5X5_100`` dictionary.

    Args:
        squares_x: Number of board squares along the horizontal axis.
        squares_y: Number of board squares along the vertical axis.
        square_mm: Board square side length, in millimeters.
        marker_mm: ArUco marker side length, in millimeters.
        dict_id: OpenCV predefined ArUco dictionary identifier.

    Returns:
        ChArUco board and its associated OpenCV detector.
    """
    aruco_dict = cv.aruco.getPredefinedDictionary(dict_id)
    board = cv.aruco.CharucoBoard(
        (squares_x, squares_y), square_mm, marker_mm, aruco_dict
    )
    detector = cv.aruco.CharucoDetector(board)
    return board, detector


def estimate_global_pose(
    rect_left_gray: np.ndarray,
    detector,
    board,
    rectified_intrinsic: np.ndarray,
    dist: np.ndarray | None = None,
    min_corners: int = 6,
):
    """Estimate the transform from the rectified left camera to the world frame.

    Detect ChArUco corners, solve the world-to-camera pose and invert it. The input
    image is assumed to be rectified, so distortion defaults to zero.

    Args:
        rect_left_gray: Rectified left grayscale image.
        detector: ChArUco detector returned by ``create_charuco_detector``.
        board: ChArUco board associated with the detector.
        rectified_intrinsic: Rectified intrinsic matrix, usually ``P1[:3, :3]``.
        dist: Distortion coefficients; ``None`` uses five zero coefficients.
        min_corners: Minimum number of detected ChArUco corners required.

    Returns:
        Homogeneous camera-to-world matrix with shape ``(4, 4)``, or ``None`` if
        too few corners are detected or pose estimation fails.
    """
    if dist is None:
        dist = np.zeros(5)

    ch_corners, ch_ids, _, _ = detector.detectBoard(rect_left_gray)
    if ch_ids is None or len(ch_ids) < min_corners:
        return None

    obj_pts, img_pts = board.matchImagePoints(ch_corners, ch_ids)

    # Uso SQPnP para encontrar el óptimo global con un costo computacional bajo y
    # admitir puntos coplanares
    ok, rvec, tvec = cv.solvePnP(
        obj_pts, img_pts, rectified_intrinsic, dist, flags=cv.SOLVEPNP_SQPNP
    )
    if not ok:
        return None

    # Ojo: solvePnP da mundo -> cámara; acá necesitamos la vuelta, cámara -> mundo.
    camera_rotation, _ = cv.Rodrigues(rvec)
    inverse_rotation = camera_rotation.T
    inverse_translation = (-inverse_rotation @ tvec).flatten()

    camera_to_world_transform = np.eye(4)
    camera_to_world_transform[:3, :3] = inverse_rotation
    camera_to_world_transform[:3, 3] = inverse_translation
    return camera_to_world_transform


def create_stereo_matcher(
    min_disparity: int = 0,
    num_disparities: int = 240,
    block_size: int = 11,
) -> cv.StereoSGBM:
    """Create the notebook's SGBM matcher for rectified grayscale pairs.

    Args:
        min_disparity: Minimum disparity searched, in pixels.
        num_disparities: Disparity search range; must be a multiple of 16.
        block_size: Matching block side length, in pixels.

    Returns:
        Matcher with the original notebook's SGBM parameters.
    """
    return cv.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 1 * block_size**2,
        P2=32 * 1 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=5,
        speckleWindowSize=200,
        speckleRange=2,
        preFilterCap=0,
        mode=cv.STEREO_SGBM_MODE_SGBM,
    )



def compute_disparities(
    images_left: Sequence[str],
    images_right: Sequence[str],
    stereo: cv.StereoSGBM,
    left_rectification_map_x: np.ndarray,
    left_rectification_map_y: np.ndarray,
    right_rectification_map_x: np.ndarray,
    right_rectification_map_y: np.ndarray,
    disparity_output_dir: str,
    use_wls: bool = True,
    lr_max_diff: float = 1.5,
    min_confidence: float = 0.0,
) -> None:
    """Compute, filter and save disparities for rectified stereo image pairs.

    Apply left-right consistency and optional WLS filtering. Save float32 pixel
    disparities as ``.npy`` files and scaled grayscale previews as ``.png`` files.
    Invalid pixels are stored as zero.

    Args:
        images_left: Left image paths, ordered to match ``images_right``.
        images_right: Right image paths, ordered to match ``images_left``.
        stereo: Left-image SGBM matcher.
        left_rectification_map_x: Horizontal remapping coordinates for the left camera.
        left_rectification_map_y: Vertical remapping coordinates for the left camera.
        right_rectification_map_x: Horizontal remapping coordinates for the right camera.
        right_rectification_map_y: Vertical remapping coordinates for the right camera.
        disparity_output_dir: Directory for disparity arrays and preview images.
        use_wls: Whether to apply weighted least-squares disparity filtering.
        lr_max_diff: Maximum left-right consistency error, in pixels.
        min_confidence: Minimum WLS confidence in the range 0-255; zero disables it.

    Raises:
        ValueError: If image-list lengths differ or filtering thresholds are invalid.
    """
    if len(images_left) != len(images_right):
        raise ValueError(
            f"Mismatched image counts: "
            f"{len(images_left)} left and {len(images_right)} right."
        )

    if lr_max_diff < 0:
        raise ValueError("lr_max_diff must be >= 0.")

    if not 0 <= min_confidence <= 255:
        raise ValueError("min_confidence must be between 0 and 255.")

    os.makedirs(disparity_output_dir, exist_ok=True)

    right_matcher = cv.ximgproc.createRightMatcher(stereo)

    wls_filter = None
    if use_wls:
        wls_filter = cv.ximgproc.createDisparityWLSFilter(
            matcher_left=stereo
        )
        wls_filter.setLambda(8000.0)
        wls_filter.setSigmaColor(1.5)

    for img_left_path, img_right_path in zip(images_left, images_right):
        left = imread_unicode(img_left_path, cv.IMREAD_GRAYSCALE)
        right = imread_unicode(img_right_path, cv.IMREAD_GRAYSCALE)

        if left is None or right is None:
            print(f"[WARN] No se pudieron leer: {img_left_path} / {img_right_path}")
            continue

        if left.shape != right.shape:
            print(f"[WARN] Tamaños distintos: {img_left_path} / {img_right_path}")
            continue

        left_rect = cv.remap(
            left,
            left_rectification_map_x,
            left_rectification_map_y,
            cv.INTER_LINEAR,
        )
        right_rect = cv.remap(
            right,
            right_rectification_map_x,
            right_rectification_map_y,
            cv.INTER_LINEAR,
        )

        # 1. Disparidades originales, en formato fijo (escala x16).
        disp_left = stereo.compute(left_rect, right_rect)
        disp_right = right_matcher.compute(right_rect, left_rect)

        # OpenCV devuelve valores por 16; primero los bajamos a píxeles reales.
        d_left = disp_left.astype(np.float32) / 16.0
        d_right = disp_right.astype(np.float32) / 16.0

        h, w = d_left.shape
        x_left = np.broadcast_to(np.arange(w, dtype=np.float32), (h, w))

        # Correspondencia: x_right = x_left - d_left.
        x_right = np.rint(x_left - d_left).astype(np.int32)

        valid_left = (
            np.isfinite(d_left)
            & (disp_left > (stereo.getMinDisparity() - 1) * 16)
            & (d_left > 0)
            & (x_right >= 0)
            & (x_right < w)
        )

        # Consultar la disparidad derecha en la posición correspondiente.
        x_right_safe = np.clip(x_right, 0, w - 1)
        rows = np.arange(h)[:, None]
        corresponding_d_right = d_right[rows, x_right_safe]

        valid_right = (
            disp_right[rows, x_right_safe]
            > (right_matcher.getMinDisparity() - 1) * 16
        )

        # La derecha viene con signo opuesto: acá se suman, no se restan.
        lr_error = np.abs(d_left + corresponding_d_right)

        lr_mask = (
            valid_left
            & valid_right
            & np.isfinite(corresponding_d_right)
            & (lr_error <= lr_max_diff)
        )

        # 2. Filtrado WLS. OpenCV espera disparidades en escala x16.
        confidence = None

        if use_wls:
            filtered_disp = wls_filter.filter(
                disp_left,
                left_rect,
                None,
                disp_right,
            )

            confidence = wls_filter.getConfidenceMap()
            disparity = filtered_disp.astype(np.float32) / 16.0
        else:
            disparity = d_left.copy()

        # 3. Máscara final: consistencia, disparidad positiva y finitud.
        valid = lr_mask & np.isfinite(disparity) & (disparity > 0)

        if confidence is not None and min_confidence > 0:
            valid &= confidence >= min_confidence

        # Convención del pipeline: 0 representa píxeles inválidos.
        disparity[~valid] = 0.0

        # 4. Guardar la disparidad real en píxeles para la reconstrucción.
        filename = os.path.basename(img_left_path)
        stem = os.path.splitext(filename)[0]
        if stem.startswith("left"):
            stem = "disp" + stem[len("left"):]

        np.save(
            os.path.join(disparity_output_dir, stem + ".npy"),
            disparity.astype(np.float32),
        )

        # 5. Vista previa: escalar usando solo píxeles válidos.
        preview = np.zeros((h, w), dtype=np.uint8)

        if np.any(valid):
            values = disparity[valid]
            lo, hi = np.percentile(values, [2, 98])

            if hi > lo:
                scaled = np.clip(
                    (values - lo) / (hi - lo), 0, 1
                )
                preview[valid] = (scaled * 255).astype(np.uint8)

        imwrite_unicode(
            os.path.join(disparity_output_dir, stem + ".png"),
            preview,
        )

        valid_ratio = 100.0 * np.count_nonzero(valid) / valid.size
        print(
            f"[OK] {stem}: {valid_ratio:.1f}% de píxeles válidos "
            f"(WLS={'sí' if use_wls else 'no'})"
        )


def merge_global_point_cloud(
    images_left: Sequence[str],
    disparity_dir: str,
    Q: np.ndarray,
    left_rectification_map_x: np.ndarray,
    left_rectification_map_y: np.ndarray,
    charuco_detector: cv.aruco.CharucoDetector,
    charuco_board: cv.aruco.CharucoBoard,
    rectified_intrinsic: np.ndarray,
    min_disparity: int = 0,
    z_min: float = 100.0,
    z_max: float = 2000.0,
    voxel_size: float = 2.0,
    roi_min: tuple[float, float, float] | None = None,
    roi_max: tuple[float, float, float] | None = None,
    merge_every: int = 1,
) -> tuple[o3d.geometry.PointCloud, list[o3d.geometry.TriangleMesh], int, int]:
    """Reconstruct and merge point clouds in the ChArUco world frame.

    Estimate each camera pose, transform its local cloud, optionally crop it and
    downsample before merging. Periodic downsampling limits accumulated cloud size.
    World coordinates and voxel dimensions use the units encoded by ``Q`` (mm here).

    Args:
        images_left: Original left image paths for the stereo captures.
        disparity_dir: Directory containing the corresponding ``disp_<n>.npy`` files.
        Q: Rectification reprojection matrix with shape ``(4, 4)``.
        left_rectification_map_x: Horizontal remapping coordinates for the left camera.
        left_rectification_map_y: Vertical remapping coordinates for the left camera.
        charuco_detector: Detector for the world-reference board.
        charuco_board: ChArUco board defining the world coordinate system.
        rectified_intrinsic: Intrinsic matrix of the rectified left camera.
        min_disparity: Exclusive lower disparity threshold, in pixels.
        z_min: Exclusive lower depth bound in the rectified camera frame.
        z_max: Exclusive upper depth bound in the rectified camera frame.
        voxel_size: Positive voxel side length for point-cloud downsampling.
        roi_min: Optional minimum XYZ bounds of the world-space region of interest.
        roi_max: Optional maximum XYZ bounds; required together with ``roi_min``.
        merge_every: Downsample the accumulated cloud after this many merged pairs.

    Returns:
        Global point cloud, camera-frame meshes, number of merged pairs and number
        of pairs discarded because their image or board pose was unavailable.
        Captures with missing disparity files are skipped without being counted.

    Raises:
        ValueError: If voxel size, merge interval or region-of-interest bounds are invalid.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive.")
    if merge_every < 1:
        raise ValueError("merge_every must be >= 1.")
    if (roi_min is None) != (roi_max is None):
        raise ValueError("Provide roi_min and roi_max together.")

    roi = None
    if roi_min is not None:
        roi_min_arr = np.asarray(roi_min, dtype=float)
        roi_max_arr = np.asarray(roi_max, dtype=float)

        if (
            roi_min_arr.shape != (3,)
            or roi_max_arr.shape != (3,)
            or not np.all(np.isfinite(roi_min_arr))
            or not np.all(np.isfinite(roi_max_arr))
            or np.any(roi_min_arr >= roi_max_arr)
        ):
            raise ValueError("ROI bounds must be valid XYZ coordinates.")

        roi = o3d.geometry.AxisAlignedBoundingBox(
            min_bound=roi_min_arr,
            max_bound=roi_max_arr,
        )

    global_point_cloud = o3d.geometry.PointCloud()
    camera_frames = []
    merged_pair_count = 0
    discarded_pair_count = 0

    for left_path in images_left:
        filename = os.path.basename(left_path)
        stem = os.path.splitext(filename)[0]
        if stem.startswith("left"):
            stem = "disp" + stem[len("left"):]

        disparity_file_path = os.path.join(disparity_dir, stem + ".npy")

        if not os.path.exists(disparity_file_path):
            continue

        disparity = np.load(disparity_file_path)

        # Detectar la pose de la cámara respecto del tablero.
        left_image_gray = imread_unicode(left_path, cv.IMREAD_GRAYSCALE)
        if left_image_gray is None:
            discarded_pair_count += 1
            continue

        rectified_left_gray = cv.remap(
            left_image_gray,
            left_rectification_map_x,
            left_rectification_map_y,
            cv.INTER_LINEAR,
        )

        camera_to_world_transform = estimate_global_pose(
            rectified_left_gray,
            charuco_detector,
            charuco_board,
            rectified_intrinsic,
        )

        if camera_to_world_transform is None:
            discarded_pair_count += 1
            continue

        # Construir la nube local y pasarla al sistema de coordenadas del mundo.
        point_cloud_i = build_local_point_cloud(
            left_path,
            disparity,
            Q,
            left_rectification_map_x,
            left_rectification_map_y,
            min_disparity=min_disparity,
            z_min=z_min,
            z_max=z_max,
        )

        point_cloud_i.transform(camera_to_world_transform)

        # 1. Recortar la región de interés en coordenadas globales.
        if roi is not None and len(point_cloud_i.points) > 0:
            point_cloud_i = point_cloud_i.crop(roi)

        # Bajamos primero la nube local, así la acumulada no se nos va de tamaño.
        if len(point_cloud_i.points) > 0:
            point_cloud_i = point_cloud_i.voxel_down_sample(voxel_size)

        # 3. Fusionar solo los puntos ya reducidos.
        global_point_cloud += point_cloud_i
        merged_pair_count += 1

        # 4. Reducir periódicamente la nube acumulada.
        if merged_pair_count % merge_every == 0:
            if len(global_point_cloud.points) > 0:
                global_point_cloud = global_point_cloud.voxel_down_sample(
                    voxel_size
                )

        # Guardar el marco de referencia de la cámara.
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=40
        )
        camera_frame.transform(camera_to_world_transform)
        camera_frames.append(camera_frame)

        print(
            f"[{merged_pair_count}] {stem}: "
            f"{len(point_cloud_i.points):,} puntos locales; "
            f"{len(global_point_cloud.points):,} puntos globales"
        )

        # Liberar referencias temporales antes de procesar el siguiente par.
        del disparity, point_cloud_i

    # Asegurar que la nube final quede reducida incluso si el último lote
    # no coincide con merge_every.
    if len(global_point_cloud.points) > 0:
        global_point_cloud = global_point_cloud.voxel_down_sample(voxel_size)

    return (
        global_point_cloud,
        camera_frames,
        merged_pair_count,
        discarded_pair_count,
    )
