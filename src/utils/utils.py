"""Utilities for loading stereo image pairs."""

from pathlib import Path

import cv2
import numpy as np


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

    img_left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    if img_left is None:
        raise OSError(f"Could not load image: {left_path}")

    img_right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
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
