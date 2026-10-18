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


def print_calibration_report_summary(calib: dict, diagnostics: dict) -> None:
    """Print rounded calibration matrices and error extrema cited in the report.

    Args:
        calib: Calibrated intrinsic, distortion, rotation and translation arrays.
        diagnostics: Calibration diagnostics containing per-capture metrics in ``rows``.
    """
    print("== Matrices con el redondeo usado en el informe ==")
    for label, key, decimals in [
        ("K_L", "left_K", 1), ("K_R", "right_K", 1),
        ("D_L", "left_dist", 5), ("D_R", "right_dist", 5), ("R", "R", 6),
    ]:
        with np.printoptions(precision=decimals, floatmode="fixed", suppress=True):
            print(f"{label} =\n{np.asarray(calib[key])}")
    print("T =", np.array2string(
        np.asarray(calib["T"]).ravel(), formatter={"float_kind": lambda v: f"{v:.6g}"}
    ))
    print(f"Baseline |t| = {np.linalg.norm(calib['T']):.2f} mm")

    rows = diagnostics["rows"]
    print("\n== Extremos de los errores por captura ==")
    for side, key in [("izquierda", "left_reprojection_rms_px"),
                      ("derecha", "right_reprojection_rms_px")]:
        worst = max(rows, key=lambda row: row[key])
        print(f"Máximo de reproyección {side}: {worst[key]:.3f} px (par {worst['pair_id']})")
    key = "vertical_alignment_rms_px"
    best = min(rows, key=lambda row: row[key])
    worst = max(rows, key=lambda row: row[key])
    print(f"Rango de alineación vertical RMS: {best[key]:.3f}--{worst[key]:.3f} px")
    print(f"Mínimo vertical: par {best['pair_id']} ({best[key]:.3f} px)")
    print("Dos mayores errores verticales:")
    for row in sorted(rows, key=lambda row: row[key], reverse=True)[:2]:
        print(f"  Par {row['pair_id']}: {row[key]:.3f} px")


def print_optimization_reduction(point_counts: dict[str, int]) -> None:
    """Print total and per-step point reductions from notebook 03's stage counts."""
    initial = point_counts["inicial"]
    final = point_counts["outliers"]
    print(f"Reducción total: {100 * (1 - final / initial):.2f}%")
    for label, before, after in [
        ("recorte AABB", "inicial", "bbox"),
        ("submuestreo por vóxeles", "bbox", "voxel"),
        ("filtro estadístico", "voxel", "outliers"),
    ]:
        removed = point_counts[before] - point_counts[after]
        print(f"Puntos eliminados por {label}: {removed:,}")


def print_measurement_crop_diagnostics(aabb, crop_min: np.ndarray, crop_max: np.ndarray) -> None:
    """Print cloud extrema and clearances to each crop face, in millimeters.

    Args:
        aabb: Axis-aligned bounding box computed from the final point cloud.
        crop_min: XYZ lower bounds used for the spatial crop.
        crop_max: XYZ upper bounds used for the spatial crop.
    """
    crop_min, crop_max = np.asarray(crop_min), np.asarray(crop_max)
    cloud_min, cloud_max = aabb.get_min_bound(), aabb.get_max_bound()
    with np.printoptions(precision=4, floatmode="fixed", suppress=True):
        print("Orden de coordenadas: (X, Y, Z); unidades: mm")
        print("Mínimo de la nube final:", cloud_min)
        print("Máximo de la nube final:", cloud_max)
        print("Límite inferior del recorte:", crop_min)
        print("Límite superior del recorte:", crop_max)
        print("Dimensiones de la caja de recorte:", crop_max - crop_min)
        print("Distancia al límite inferior (mínimo nube - recorte):", cloud_min - crop_min)
        print("Distancia al límite superior (recorte - máximo nube):", crop_max - cloud_max)


def print_pose_diagnostics(diagnostics: dict) -> None:
    """Print pose acceptance, reprojection errors and camera coordinates for the report.

    Args:
        diagnostics: Accepted IDs, camera-to-world poses, rejected IDs and per-pose RMS.
    """
    pair_ids = diagnostics["pair_ids"]
    rejected = diagnostics["rejected_pair_ids"]
    poses = np.asarray(diagnostics["poses"])
    errors = np.asarray(diagnostics["reprojection_rms_px"])
    total = len(pair_ids) + len(rejected)
    print(f"Poses válidas: {len(pair_ids)} / {total} ({100 * len(pair_ids) / total:.1f}%)")
    print(f"Pares descartados: {rejected}")
    print(f"RMS PnP: mediana {np.median(errors):.2f} px; máximo {np.max(errors):.2f} px")
    print(f"Mayor RMS: par {pair_ids[int(np.argmax(errors))]} ({np.max(errors):.3f} px)")
    centers = poses[:, :3, 3]
    print(f"Centro de la cámara del par {pair_ids[0]} (mm): {np.round(centers[0], 1)}")
    print(f"Centro redondeado a mm enteros: {np.rint(centers[0]).astype(int)}")
    print(f"Rango de Z de las cámaras (mm): {np.min(centers[:, 2]):.1f} a {np.max(centers[:, 2]):.1f}")
