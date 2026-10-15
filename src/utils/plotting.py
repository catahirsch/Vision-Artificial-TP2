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


def _guardar(fig, guardar: str | None) -> None:
    """Guarda ``fig`` en ``guardar`` creando el directorio si no existe."""
    if guardar:
        carpeta = os.path.dirname(guardar)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        fig.savefig(guardar, dpi=120, bbox_inches="tight")


def mostrar_par_rectificado(
    img_left: np.ndarray,
    img_right: np.ndarray,
    rect_left: np.ndarray,
    rect_right: np.ndarray,
    image_size: tuple[int, int],
    n_lineas: int = 10,
    guardar: str | None = None,
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
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].imshow(cv.cvtColor(img_left, cv.COLOR_BGR2RGB));   axs[0, 0].set_title("Izquierda Original")
    axs[0, 1].imshow(cv.cvtColor(img_right, cv.COLOR_BGR2RGB));  axs[0, 1].set_title("Derecha Original")
    axs[1, 0].imshow(cv.cvtColor(rect_left, cv.COLOR_BGR2RGB));  axs[1, 0].set_title("Izquierda Rectificada")
    axs[1, 1].imshow(cv.cvtColor(rect_right, cv.COLOR_BGR2RGB)); axs[1, 1].set_title("Derecha Rectificada")

    for y in np.linspace(0, image_size[1], n_lineas)[1:-1]:
        axs[1, 0].axhline(y, color="r", linestyle="--", alpha=0.6)
        axs[1, 1].axhline(y, color="r", linestyle="--", alpha=0.6)

    for ax in axs.ravel():
        ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    _guardar(fig, guardar)
    return fig


def mostrar_disparidad(
    disparity: np.ndarray,
    titulo: str = "Mapa de disparidad",
    cmap: str = "magma",
    guardar: str | None = None,
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
    fig = plt.figure(figsize=(10, 5))
    plt.imshow(disparity, cmap=cmap)
    plt.title(titulo)
    plt.colorbar(label="disparidad (px)")
    plt.axis("off")
    _guardar(fig, guardar)
    return fig


def visualizar_nubes(geometrias: list, titulo: str = "Open3D"):
    """
    Abre una ventana interactiva de Open3D con las geometrías dadas.

    Args:
        geometrias (list): Lista de geometrías de Open3D (nubes, mallas, ejes).
        titulo (str): Título de la ventana.
    """
    o3d.visualization.draw_geometries(geometrias, window_name=titulo)


def guardar_captura_o3d(
    geometrias: list,
    guardar: str,
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
    carpeta = os.path.dirname(guardar)
    if carpeta:
        os.makedirs(carpeta, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False, width=width, height=height)
    for g in geometrias:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(guardar, do_render=True)
    vis.destroy_window()
