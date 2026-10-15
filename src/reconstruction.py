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

import numpy as np
import cv2 as cv
import open3d as o3d


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
    left_image = cv.imread(left_path, cv.IMREAD_COLOR)
    rectified_left_color = cv.remap(
        left_image,
        left_rectification_map_x,
        left_rectification_map_y,
        cv.INTER_LINEAR,
    )

    points_3d = cv.reprojectImageTo3D(disparity, Q)  # (H, W, 3) en mm
    colors = cv.cvtColor(rectified_left_color, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

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
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, rectified_intrinsic, dist)
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
