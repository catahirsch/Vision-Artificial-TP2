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
        name (str): Label displayed to the left of the matrix.
        matrix (np.ndarray): Two-dimensional array containing the values to display.

    Raises:
        ValueError: If the array does not have exactly two dimensions.
    """
    if matrix.ndim != 2:
        raise ValueError("matrix debe ser un array de dos dimensiones.")

    rows = []
    for row in matrix:
        cells = []
        for value in row:
            text = f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
            cells.append(
                '<td style="border: none; padding: 4px 12px; text-align: right;">'
                f"{escape(text)}</td>"
            )
        rows.append(f"<tr>{''.join(cells)}</tr>")

    display(HTML(
        '<div style="display: flex; align-items: center; gap: 12px; '
        'margin: 12px 0; overflow-x: auto;">'
        f'<strong style="white-space: nowrap;">{escape(str(name))} =</strong>'
        '<div style="border-left: 2px solid currentColor; '
        'border-right: 2px solid currentColor; padding: 4px;">'
        '<table style="border-collapse: collapse; font-family: monospace; '
        'font-variant-numeric: tabular-nums; white-space: nowrap;">'
        f"{''.join(rows)}</table></div></div>"
    ))
