"""Utilities for loading stereo image pairs."""

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Lee una imagen de forma segura ante rutas con caracteres no-ASCII.

    ``cv2.imread`` falla silenciosamente (devuelve ``None``) cuando la ruta
    contiene tildes, ``ñ`` o ``°`` en Windows. Esta función lee los bytes con
    ``np.fromfile`` (que respeta Unicode) y los decodifica con ``cv2.imdecode``.

    Args:
        path: Ruta de la imagen.
        flags: Modo de lectura de OpenCV (por defecto, color BGR).

    Returns:
        La imagen como ``np.ndarray``, o ``None`` si no se pudo leer/decodificar.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Escribe una imagen de forma segura ante rutas con caracteres no-ASCII.

    Análogo de ``imread_unicode`` para la escritura: codifica con
    ``cv2.imencode`` según la extensión y vuelca los bytes con ``ndarray.tofile``,
    que respeta Unicode en Windows (a diferencia de ``cv2.imwrite``).

    Args:
        path: Ruta de salida (la extensión define el formato, p. ej. ``.png``).
        image: Imagen a guardar.

    Returns:
        ``True`` si se escribió correctamente, ``False`` en caso contrario.
    """
    path = Path(path)
    ext = path.suffix or ".png"
    ok, buffer = cv2.imencode(ext, image)
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


def get_image_at_index(dataset_name: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    """Load a stereo image pair from a dataset in ``data/raw``.

    Args:
        dataset_name: Dataset directory name, such as ``palacio``.
        index: Number in the ``left_<index>.jpg`` and ``right_<index>.jpg`` names.

    Returns:
        Left and right color images in OpenCV's BGR format.

    Raises:
        OSError: If either image cannot be loaded.
    """
    dataset_dir = Path(__file__).resolve().parents[2] / "data" / "raw" / dataset_name
    left_path = dataset_dir / f"left_{index}.jpg"
    right_path = dataset_dir / f"right_{index}.jpg"

    img_left = imread_unicode(left_path, cv2.IMREAD_COLOR)
    if img_left is None:
        raise OSError(f"Could not load image: {left_path}")

    img_right = imread_unicode(right_path, cv2.IMREAD_COLOR)
    if img_right is None:
        raise OSError(f"Could not load image: {right_path}")

    return img_left, img_right


def indexed_images(directory: str | Path, side: str) -> dict[int, Path]:
    """Index image paths by capture number, rejecting duplicate indices."""
    paths = list(Path(directory).glob(f"{side}_*.jpg"))
    indexed = {int(path.stem.rsplit("_", 1)[1]): path for path in paths}
    if len(indexed) != len(paths):
        raise ValueError(f"Hay índices de captura repetidos para {side}.")
    return indexed
