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


def indice_par(path: str) -> int:
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


def construir_nube_local(
    left_path: str,
    disparity: np.ndarray,
    Q: np.ndarray,
    map1x: np.ndarray,
    map1y: np.ndarray,
    min_disp: int = 0,
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
    rectL_color = cv.remap(cv.imread(left_path, cv.IMREAD_COLOR), map1x, map1y, cv.INTER_LINEAR)

    pts_3d = cv.reprojectImageTo3D(disparity, Q)  # (H, W, 3) en mm
    colors = cv.cvtColor(rectL_color, cv.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Filtrado: disparidad válida, coordenadas finitas y profundidad en rango
    mask = (disparity > min_disp) & np.isfinite(pts_3d).all(axis=2)
    mask &= (pts_3d[:, :, 2] > z_min) & (pts_3d[:, :, 2] < z_max)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_3d[mask])
    pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    return pcd


def crear_detector_charuco(
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
    board = cv.aruco.CharucoBoard((squares_x, squares_y), square_mm, marker_mm, aruco_dict)
    detector = cv.aruco.CharucoDetector(board)
    return board, detector


def estimar_pose_global(
    rect_left_gray: np.ndarray,
    detector,
    board,
    K_rect: np.ndarray,
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
    ok, rvec, tvec = cv.solvePnP(obj_pts, img_pts, K_rect, dist)
    if not ok:
        return None

    # solvePnP da mundo -> cámara; se invierte para llevar la nube de cámara a mundo.
    R_cam, _ = cv.Rodrigues(rvec)
    R_inv = R_cam.T
    t_inv = (-R_inv @ tvec).flatten()

    T_global = np.eye(4)
    T_global[:3, :3] = R_inv
    T_global[:3, 3] = t_inv
    return T_global
