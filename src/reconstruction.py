"""
Funciones del pipeline de reconstrucción 3D por estéreo multivista (TP2).

Agrupa las utilidades usadas en ``notebooks/02_reconstruction.ipynb``:

- ``indice_par``: ordenamiento numérico de los pares de imágenes.
- ``construir_nube_local``: nube de puntos 3D de un par, a partir de su disparidad.
- ``crear_detector_charuco`` / ``estimar_pose_global``: pose de la cámara respecto
  del tablero ChArUco (marco de mundo).

Las funciones reciben explícitamente sus dependencias (mapas de rectificación,
matriz ``Q``, intrínseca rectificada, etc.) para no depender de variables globales
del notebook.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import cv2 as cv
import open3d as o3d

from .utils.utils import imread_unicode, imwrite_unicode


def get_pair_index(path: str) -> int:
    """
    Índice numérico de una imagen ``left_<n>.jpg`` / ``right_<n>.jpg``.

    Permite ordenar los pares por número (0, 1, 2, ...) en lugar de
    alfabéticamente (0, 1, 10, 11, 2, ...).

    Args:
        path (str): Ruta o nombre de archivo de la imagen.

    Returns:
        int: El número ``<n>`` contenido en el nombre del archivo.
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
    """
    Nube de puntos 3D (frame rectificado izquierdo) a partir de la disparidad y Q.

    Reproyecta la disparidad a 3D con ``cv.reprojectImageTo3D`` y la matriz ``Q``,
    colorea cada punto con la imagen izquierda rectificada y descarta los píxeles
    sin correspondencia válida y los que caen fuera del rango de profundidad.

    Args:
        left_path (str): Ruta de la imagen izquierda original del par.
        disparity (np.ndarray): Mapa de disparidad real (float, en píxeles).
        Q (np.ndarray): Matriz de reproyección 4x4 obtenida en la rectificación.
        map1x, map1y (np.ndarray): Mapas de rectificación de la cámara izquierda.
        min_disp (int): Disparidad mínima válida; por debajo se descarta.
        z_min, z_max (float): Rango de profundidad válido en milímetros.

    Returns:
        o3d.geometry.PointCloud: Nube de puntos coloreada en el sistema de la
        cámara izquierda rectificada.
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
    """
    Construye el tablero ChArUco del marco de mundo y su detector.

    Los valores por defecto corresponden al tablero usado en las capturas de
    reconstrucción (grid 9x13, cuadrado 42 mm, marcador 31.5 mm, DICT_5X5_100).

    Returns:
        tuple: ``(board, detector)`` de OpenCV (``cv.aruco.CharucoBoard`` y
        ``cv.aruco.CharucoDetector``).
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
    """
    Transformación 4x4 (cámara rectificada -> mundo), o ``None`` si no hay tablero.

    Detecta el tablero ChArUco en la imagen rectificada, estima la pose
    mundo -> cámara con ``cv.solvePnP`` y la invierte para obtener cámara -> mundo.
    La imagen se asume rectificada, por lo que se usa la intrínseca rectificada y
    distorsión nula.

    Args:
        rect_left_gray (np.ndarray): Imagen izquierda rectificada en escala de grises.
        detector: ``cv.aruco.CharucoDetector`` (ver ``crear_detector_charuco``).
        board: ``cv.aruco.CharucoBoard`` asociado al detector.
        K_rect (np.ndarray): Intrínseca rectificada (``P1[:3, :3]``).
        dist (np.ndarray | None): Distorsión; por defecto nula (imagen rectificada).
        min_corners (int): Mínimo de esquinas ChArUco para aceptar la detección.

    Returns:
        np.ndarray | None: Matriz homogénea 4x4 cámara -> mundo, o ``None``.
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

    # solvePnP da mundo -> cámara; se invierte para llevar la nube de cámara a mundo.
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
) -> None:
    """Rectify image pairs and save their disparities using the notebook's loop.

    Args:
        images_left: Left image paths, ordered to match ``images_right``.
        images_right: Corresponding right image paths.
        stereo: Matcher returned by ``create_stereo_matcher``.
        left_rectification_map_x, left_rectification_map_y: Left rectification maps.
        right_rectification_map_x, right_rectification_map_y: Right rectification maps.
        disparity_output_dir: Existing directory for NPY disparities and PNG previews.

    NPY files contain real float32 disparities in pixels for reconstruction.
    PNG files are normalized previews. Pairs with unreadable images are skipped.
    """
    for img_left_path, img_right_path in zip(images_left, images_right):
        left_image_gray = imread_unicode(img_left_path, cv.IMREAD_GRAYSCALE)
        right_image_gray = imread_unicode(img_right_path, cv.IMREAD_GRAYSCALE)
        if left_image_gray is None or right_image_gray is None:
            continue

        rectified_left_gray = cv.remap(
            left_image_gray,
            left_rectification_map_x,
            left_rectification_map_y,
            cv.INTER_LINEAR,
        )
        rectified_right_gray = cv.remap(
            right_image_gray,
            right_rectification_map_x,
            right_rectification_map_y,
            cv.INTER_LINEAR,
        )

        # SGBM devuelve la disparidad en enteros escalados x16 -> pasar a float real
        disparity = (
            stereo.compute(rectified_left_gray, rectified_right_gray).astype(np.float32)
            / 16.0
        )

        disparity_filename_stem = (
            os.path.basename(img_left_path).replace("left", "disp").replace(".jpg", "")
        )
        # Disparidad REAL (float): es la que usa la nube local. NO usar el PNG para 3D.
        np.save(
            os.path.join(disparity_output_dir, disparity_filename_stem + ".npy"),
            disparity,
        )
        # PNG normalizado solo para visualización / informe
        disparity_preview = cv.normalize(
            disparity, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U
        )
        imwrite_unicode(
            os.path.join(disparity_output_dir, disparity_filename_stem + ".png"),
            disparity_preview,
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
) -> tuple[o3d.geometry.PointCloud, list[o3d.geometry.TriangleMesh], int, int]:
    """Transform and merge local clouds using the notebook's reconstruction loop.

    Args:
        images_left: Left image paths in capture order.
        disparity_dir: Directory containing real ``disp_<index>.npy`` disparities.
        Q: Disparity-to-3D reprojection matrix from stereo rectification.
        left_rectification_map_x, left_rectification_map_y: Left rectification maps.
        charuco_detector: Detector returned by ``create_charuco_detector``.
        charuco_board: ChArUco board associated with the detector.
        rectified_intrinsic: Rectified left intrinsic matrix, ``P1[:3, :3]``.
        min_disparity: Minimum valid disparity for local point clouds.
        z_min, z_max: Valid depth range in millimeters in each local camera frame.

    Returns:
        Merged cloud, camera coordinate frames, merged pair count and count
        of pairs discarded because their board pose was not found.
        Images without a saved disparity are skipped.
    """
    global_point_cloud = o3d.geometry.PointCloud()
    camera_frames = []  # poses de cámara para visualizar
    merged_pair_count, discarded_pair_count = 0, 0

    for left_path in images_left:
        disparity_filename_stem = (
            os.path.basename(left_path).replace("left", "disp").replace(".jpg", "")
        )
        disparity_file_path = os.path.join(
            disparity_dir, disparity_filename_stem + ".npy"
        )
        if not os.path.exists(disparity_file_path):
            continue
        disparity = np.load(disparity_file_path)

        # Pose de esta cámara respecto del mundo (tablero ChArUco)
        left_image_gray = imread_unicode(left_path, cv.IMREAD_GRAYSCALE)
        rectified_left_gray = cv.remap(
            left_image_gray,
            left_rectification_map_x,
            left_rectification_map_y,
            cv.INTER_LINEAR,
        )
        camera_to_world_transform = estimate_global_pose(
            rectified_left_gray, charuco_detector, charuco_board, rectified_intrinsic
        )
        if (
            camera_to_world_transform is None
        ):  # sin tablero detectado -> se descarta el par
            discarded_pair_count += 1
            continue

        # Nube local y TRANSFORMACIÓN cámara -> mundo
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
        global_point_cloud += point_cloud_i

        # Pose de la cámara (marco de ejes) para verificar la órbita de captura
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=40)
        camera_frame.transform(camera_to_world_transform)
        camera_frames.append(camera_frame)
        merged_pair_count += 1
    return global_point_cloud, camera_frames, merged_pair_count, discarded_pair_count
