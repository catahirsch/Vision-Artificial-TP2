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
    use_wls: bool = True,
    lr_max_diff: float = 1.5,
    min_confidence: float = 0.0,
) -> None:
    """
    Calcula disparidades, filtra con WLS opcional y aplica consistencia
    izquierda-derecha. Guarda disparidades float32 en píxeles.

    Args:
        images_left, images_right: Pares correspondientes y en el mismo orden.
        stereo: Matcher SGBM para la imagen izquierda.
        *_rectification_map_*: Mapas de rectificación de ambas cámaras.
        disparity_output_dir: Directorio de salida.
        use_wls: Si True, aplica filtrado WLS.
        lr_max_diff: Error máximo permitido en la consistencia L-R, en píxeles.
        min_confidence: Confianza WLS mínima (0-255). 0 desactiva este filtro.
    """
    if len(images_left) != len(images_right):
        raise ValueError(
            f"Cantidad de imágenes distinta: "
            f"{len(images_left)} izquierdas y {len(images_right)} derechas."
        )

    if lr_max_diff < 0:
        raise ValueError("lr_max_diff debe ser >= 0.")

    if not 0 <= min_confidence <= 255:
        raise ValueError("min_confidence debe estar entre 0 y 255.")

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

        # Convertir a píxeles reales para evaluar la consistencia.
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

        # La disparidad derecha tiene signo opuesto.
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
    """
    Reconstruye y fusiona nubes de puntos reduciendo su tamaño progresivamente.

    roi_min y roi_max definen los límites XYZ de una ROI en coordenadas
    del mundo, en las mismas unidades que Q (en este pipeline, mm).

    voxel_size: lado del vóxel para downsampling, en unidades del mundo.
    merge_every: reduce la nube global cada N pares procesados.
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size debe ser positivo.")
    if merge_every < 1:
        raise ValueError("merge_every debe ser >= 1.")
    if (roi_min is None) != (roi_max is None):
        raise ValueError("Hay que proporcionar roi_min y roi_max juntos.")

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
            raise ValueError("Los límites de la ROI deben ser XYZ válidos.")

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

        # 2. Reducir la nube local ANTES de fusionarla.
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
