"""
Funciones de graficación y visualización del TP2.

Encapsula los gráficos usados en los notebooks para poder generarlos con una sola
llamada:

- ``mostrar_par_rectificado``: par original vs. rectificado con líneas epipolares.
- ``mostrar_disparidad``: mapa de disparidad con barra de color.
- ``visualizar_nubes``: ventana interactiva de Open3D.
- ``guardar_captura_o3d``: render *offscreen* de una escena de Open3D a PNG.

Las funciones de Matplotlib aceptan un parámetro ``guardar`` con la ruta de salida;
si se indica, la figura se guarda (creando el directorio si hace falta).
"""

from __future__ import annotations

import os

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import open3d as o3d

from .utils import get_image_at_index
from ..calibration_rectification import rectify_image_pair


def _save_figure(figure, save_path: str | None) -> None:
    """Guarda ``fig`` en ``guardar`` creando el directorio si no existe."""
    if save_path:
        directory = os.path.dirname(save_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        figure.savefig(save_path, dpi=120, bbox_inches="tight")


def show_rectified_pair(
    img_left: np.ndarray,
    img_right: np.ndarray,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    image_size: tuple[int, int],
    num_lines: int = 10,
    save_path: str | None = None,
):
    """
    Muestra el par original vs. rectificado en una grilla 2x2.

    Sobre las imágenes rectificadas dibuja líneas horizontales de referencia: un
    mismo punto físico debe caer en la misma fila en ambas, lo que verifica la
    rectificación. Las imágenes se asumen en formato BGR (como las lee OpenCV).

    Args:
        img_left, img_right: Par original (BGR).
        rect_left, rect_right: Par rectificado (BGR).
        image_size (tuple): ``(ancho, alto)`` en píxeles.
        n_lineas (int): Cantidad de líneas epipolares de referencia.
        guardar (str | None): Ruta de salida del PNG; si es None, no se guarda.

    Returns:
        matplotlib.figure.Figure: La figura generada.
    """
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].imshow(cv.cvtColor(img_left, cv.COLOR_BGR2RGB))
    axes[0, 0].set_title("Izquierda Original")
    axes[0, 1].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB))
    axes[0, 1].set_title("Derecha Original")
    axes[1, 0].imshow(cv.cvtColor(rect_left, cv.COLOR_BGR2RGB))
    axes[1, 0].set_title("Izquierda Rectificada")
    axes[1, 1].imshow(cv.cvtColor(rect_right, cv.COLOR_BGR2RGB))
    axes[1, 1].set_title("Derecha Rectificada")

    for y in np.linspace(0, image_size[1], num_lines)[1:-1]:
        axes[1, 0].axhline(y, color="r", linestyle="--", alpha=0.6)
        axes[1, 1].axhline(y, color="r", linestyle="--", alpha=0.6)

    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])

    figure.tight_layout()
    _save_figure(figure, save_path)


def show_disparity(
    disparity: np.ndarray,
    title: str = "Mapa de disparidad",
    cmap: str = "magma",
    save_path: str | None = None,
):
    """
    Muestra un mapa de disparidad con barra de color.

    Args:
        disparity (np.ndarray): Mapa de disparidad (float, en píxeles).
        titulo (str): Título de la figura.
        cmap (str): Mapa de color de Matplotlib.
        guardar (str | None): Ruta de salida del PNG; si es None, no se guarda.

    Returns:
        matplotlib.figure.Figure: La figura generada.
    """
    figure = plt.figure(figsize=(10, 5))
    plt.imshow(disparity, cmap=cmap)
    plt.title(title)
    plt.colorbar(label="disparidad (px)")
    plt.axis("off")
    _save_figure(figure, save_path)
    plt.show()


def show_geometries(geometries: list, title: str = "Open3D"):
    """
    Abre una ventana interactiva de Open3D con las geometrías dadas.

    Args:
        geometrias (list): Lista de geometrías de Open3D (nubes, mallas, ejes).
        titulo (str): Título de la ventana.
    """
    o3d.visualization.draw_geometries(geometries, window_name=title)


def save_open3d_screenshot(
    geometries: list,
    save_path: str,
    width: int = 1280,
    height: int = 960,
) -> None:
    """
    Renderiza una escena de Open3D a una imagen PNG sin abrir una ventana visible.

    Útil para generar figuras del informe (por ejemplo, la nube global con las
    poses de cámara) de forma automática.

    Args:
        geometrias (list): Lista de geometrías de Open3D a renderizar.
        guardar (str): Ruta de salida del PNG.
        width, height (int): Resolución de la captura.
    """
    directory = os.path.dirname(save_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(visible=False, width=width, height=height)
    for g in geometries:
        visualizer.add_geometry(g)
    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(save_path, do_render=True)
    visualizer.destroy_window()


def show_calibration_errors(diagnostics: dict, save_path: str | None = None):
    """Plot, optionally save and display reprojection and vertical errors per capture."""
    rows = diagnostics["rows"]
    used_pair_ids = [row["pair_id"] for row in rows]
    per_view_rms = np.array(
        [
            [row["left_reprojection_rms_px"], row["right_reprojection_rms_px"]]
            for row in rows
        ]
    )
    diagnostics = rows
    figure, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].plot(used_pair_ids, per_view_rms[:, 0], ".-", label="Izquierda")
    axes[0].plot(
        used_pair_ids, per_view_rms[:, 1], ".-", label="Derecha desde pose izquierda"
    )
    axes[0].set_ylabel("RMS reproyección (px)")
    axes[0].legend()
    axes[1].plot(
        used_pair_ids, [row["vertical_alignment_rms_px"] for row in diagnostics], ".-"
    )
    axes[1].set_ylabel("RMS vertical (px)")
    axes[1].set_xlabel("Índice del par de calibración")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    _save_figure(figure, save_path)
    plt.show()


def show_rectification_example(
    dataset_name: str,
    index: int,
    maps: dict,
    image_size: tuple[int, int],
    save_path: str | None = None,
):
    """Load a dataset pair, rectify it and display/save the comparison figure."""
    img_left, img_right = get_image_at_index(dataset_name=dataset_name, index=index)
    if img_left.shape[1::-1] != image_size or img_right.shape[1::-1] != image_size:
        raise ValueError(
            "La resolución del ejemplo difiere de la resolución de calibración."
        )
    rect_left, rect_right = rectify_image_pair(img_left, img_right, maps)
    figure = show_rectified_pair(
        img_left,
        img_right,
        rect_left,
        rect_right,
        image_size,
        save_path=save_path,
    )
    plt.show()
