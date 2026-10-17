"""Utilities for loading stereo image pairs."""

from pathlib import Path

import cv2
import numpy as np


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read an image safely from a path containing non-ASCII characters.

    On Windows, ``cv2.imread`` may silently return ``None`` for Unicode paths.
    Read bytes with ``np.fromfile`` and decode them with ``cv2.imdecode`` instead.

    Args:
        path: Input image path.
        flags: OpenCV decoding mode; defaults to a BGR color image.

    Returns:
        Decoded image array, or ``None`` if the file cannot be read or decoded.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Write an image safely to a path containing non-ASCII characters.

    Encode with ``cv2.imencode`` and write the bytes with ``ndarray.tofile`` to
    support Unicode paths on Windows.

    Args:
        path: Output path; its extension selects the format, defaulting to PNG.
        image: Image array to encode and save.

    Returns:
        ``True`` if encoding and writing succeed, or ``False`` if encoding fails.
        File-writing errors propagate to the caller.
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
    """Index stereo image paths by capture number, rejecting duplicate indices.

    Args:
        directory: Directory containing stereo image files.
        side: Filename prefix identifying the camera, usually ``left`` or ``right``.

    Returns:
        Mapping from capture numbers to matching ``<side>_<index>.jpg`` paths.

    Raises:
        ValueError: If a filename has a non-numeric index or indices are duplicated.
    """
    paths = list(Path(directory).glob(f"{side}_*.jpg"))
    indexed = {int(path.stem.rsplit("_", 1)[1]): path for path in paths}
    if len(indexed) != len(paths):
        raise ValueError(f"Duplicate capture indices for {side}.")
    return indexed
