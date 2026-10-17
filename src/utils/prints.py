"""
Formatting utilities for displaying NumPy matrices in Jupyter notebooks.

Provides HTML output with matrix labels, brackets, aligned columns, and compact
numeric formatting through IPython's rich display system.
"""

from html import escape

import numpy as np
from IPython.display import HTML, display


def print_matrix(name: str, matrix: np.ndarray) -> None:
    """
    Display a named 2D matrix as formatted HTML in a Jupyter notebook.

    The matrix is enclosed in brackets with right-aligned columns. Floating-point
    values are formatted with up to six significant digits. The input array is
    not modified.

    Args:
        name: Label displayed to the left of the matrix.
        matrix: Two-dimensional array containing the values to display.

    Raises:
        ValueError: If the array does not have exactly two dimensions.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix must be a two-dimensional array.")

    rows = []
    for row in matrix:
        cells = []
        for value in row:
            text = (
                f"{value:.6g}"
                if isinstance(value, (float, np.floating))
                else str(value)
            )
            cells.append(
                '<td style="border: none; padding: 4px 12px; text-align: right;">'
                f"{escape(text)}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    display(
        HTML(
            '<div style="display: flex; align-items: center; gap: 12px; '
            'margin: 12px 0; overflow-x: auto;">'
            f'<strong style="white-space: nowrap;">{escape(str(name))} =</strong>'
            '<div style="border-left: 2px solid currentColor; '
            'border-right: 2px solid currentColor; padding: 4px;">'
            '<table style="border-collapse: collapse; font-family: monospace; '
            'font-variant-numeric: tabular-nums; white-space: nowrap;">'
            f"{''.join(rows)}</table></div></div>"
        )
    )


def print_calibration_results(calib: dict, calibration_rms: dict) -> None:
    """Display calibrated matrices, RMS values and the metric baseline.

    Args:
        calib: Stereo calibration matrices and image size.
        calibration_rms: Left, right and stereo calibration RMS values, in pixels.
    """
    for name, value in calib.items():
        if name != "image_size":
            print_matrix(name=name, matrix=value)

    print(f"RMS calibración monocular izquierda: {calibration_rms['left']:.3f} px")
    print(f"RMS calibración monocular derecha: {calibration_rms['right']:.3f} px")
    print(f"RMS calibración estéreo (OpenCV): {calibration_rms['stereo']:.3f} px")
    print(f"Baseline |t| = {np.linalg.norm(calib['T']):.2f} mm")


def print_calibration_diagnostics(
    diagnostics: dict, rectified_focal: float, alpha: float
) -> None:
    """Print validation errors and the five worst right-camera views.

    Args:
        diagnostics: Aggregate validation errors and per-capture metrics in ``rows``.
        rectified_focal: Rectified focal length, in pixels.
        alpha: Rectification scaling used to generate the maps.
    """
    validation_rms = diagnostics["validation_rms"]
    vertical_rms = diagnostics["vertical_rms"]
    diagnostics = diagnostics["rows"]
    print(f"Cantidad de pares guardados: {len(diagnostics)}")
    print(f"RMS de validación estéreo con pose izquierda: {validation_rms:.3f} px")
    print(f"Desalineación vertical rectificada RMS: {vertical_rms:.3f} px")
    print(f"Focal rectificada: {rectified_focal:.2f} px | alpha = {alpha}")
    print("\nCinco pares con mayor error de reproyección derecha:")
    print("  Par | Izquierda (px) | Derecha (px) | Vertical (px)")
    for row in sorted(
        diagnostics, key=lambda row: row["right_reprojection_rms_px"], reverse=True
    )[:5]:
        print(
            f"{row['pair_id']:5d} | {row['left_reprojection_rms_px']:14.3f} | "
            f"{row['right_reprojection_rms_px']:12.3f} | {row['vertical_alignment_rms_px']:13.3f}"
        )
