# TP2 — Reconstrucción 3D por estéreo multivista

I308 Visión Artificial — Universidad de San Andrés
Grupo: Agustín Patruno Carranza · Catalina Hirsch · Mathias Kneipp

Pipeline de fotogrametría (estéreo multivista) para reconstruir la nube de
puntos 3D de un objeto y estimar sus dimensiones métricas reales, usando
**OpenCV** y **Open3D**.

## Entregable final

- **[Descargar la nube del Palacio optimizada (`.ply` binario)](data/processed/palacio/nube_palacio_optimizada.ply?raw=true)**:
  5,47 MB y 202.741 puntos en la ejecución actual.
- **[Informe PDF](report/patruno_hirsch_kneipp-tp2-informe.pdf)**.
- **Datasets propios incluidos:** [calibración](data/calibration) y
  [Palacio](data/raw/palacio).

La nube final se conserva en su ubicación original, sin duplicarla en otra
carpeta. `.gitignore` permite incluir únicamente ese `.ply`; la nube global y las
disparidades siguen excluidas. En GitHub también se puede usar **Download raw
file** desde la página del archivo para descargarlo sin bajar todo el proyecto.

## Estructura del repositorio

```
.
├── data/                  # fotografías y resultados de calibración
│   ├── calibration/       # 63 pares del tablero de calibración
│   ├── calibration_results/ # .pkl, matrices legibles y metadatos
│   ├── raw/               # palacio (72 pares estéreo del objeto)
│   └── processed/palacio/ # nube optimizada incluida; intermedios excluidos
├── notebooks/             # calibración, reconstrucción y optimización/medición
├── src/                   # código fuente del pipeline (ver abajo)
├── report/                # informe (LaTeX + PDF)
└── README.md
```

## Requisitos

- Entorno probado: **Python 3.13.7**.
- Versiones exactas de todas las dependencias: [requirements.txt](requirements.txt).
  Dependencias principales:

```
opencv-contrib-python==4.13.0.92
open3d==0.20.0
numpy==2.2.6
```

Instalación:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

En Windows, activar con `.venv\Scripts\activate`. Abrir los notebooks en VS Code
con las extensiones Python y Jupyter y seleccionar el intérprete de `.venv`.
No instalar `opencv-python` junto a `opencv-contrib-python`, que ya incluye los
módulos principales y ArUco.

## Uso

La calibración se realiza en
[`notebooks/01_calibration.ipynb`](notebooks/01_calibration.ipynb), a partir de los
pares de `data/calibration`. El notebook estima los intrínsecos y la distorsión de
ambas cámaras, calcula los extrínsecos estéreo y genera la rectificación con
`alpha=0`. Guarda `stereo_calibration.pkl`, `stereo_maps.pkl`, los metadatos y un
diagnóstico por captura en [`data/calibration_results`](data/calibration_results).
El notebook 02 carga esos archivos para reconstruir.

La configuración usa un checkerboard de **9 × 6 esquinas internas** (10 × 7
cuadrados) y cuadrados de **28 mm de lado**: un patrón de **280 × 196 mm**, sin
márgenes. Confirmar el lado físico del tablero y validar una distancia conocida
antes de reportar medidas en milímetros. Los RMS y la alineación rectificada
deben revisarse antes de aceptar la calibración.

El campo `metric_scale.physical_square_size_confirmed` de `metadata.yaml` se
guarda automáticamente como `True` en la implementación actual; ese valor no
acredita una comprobación física del patrón. La escala debe validarse midiendo
el lado del cuadrado impreso y contrastando una distancia conocida con la
reconstrucción. Hasta realizar esa validación, las medidas en milímetros deben
considerarse provisionales.

Ejecutar los notebooks en orden y de arriba abajo. Las rutas funcionan con el
kernel iniciado desde la raíz o desde `notebooks/`.

1. **Calibración y rectificación estéreo**
   Abrir `notebooks/01_calibration.ipynb`, seleccionar el entorno `.venv` y ejecutar
   todas las celdas. El kernel puede iniciarse desde la raíz o desde `notebooks/`.
   Cada ejecución actualiza los parámetros, mapas y diagnósticos guardados.
   Después de recalibrar, ejecutar el notebook 02 completo para regenerar las
   disparidades y nubes con los nuevos mapas y `Q`.
2. **Disparidad + nube local + pose global**
   Ejecutar [notebooks/02_reconstruction.ipynb](notebooks/02_reconstruction.ipynb).
   La configuración predeterminada usa `dataset_name = "palacio"`. Carga los mapas
   del notebook 01, genera disparidades con SGBM, WLS y consistencia
   izquierda–derecha, y estima las poses respecto del ChArUco. Guarda los
   intermedios en `data/processed/palacio/`, incluida
   `nube_global_palacio.ply`. El ChArUco usa una grilla de 9 × 13 cuadrados de
   42 mm, marcadores de 31,5 mm y `DICT_5X5_100`.
3. **Fusión y optimización de la nube + export `.ply`**
   La fusión se realiza al final del notebook 02. Luego ejecutar
   [notebooks/03_optimization_measurement.ipynb](notebooks/03_optimization_measurement.ipynb):
   carga la nube global, aplica el recorte espacial, submuestreo de 1,5 mm y
   eliminación estadística de outliers (`nb_neighbors=20`, `std_ratio=1.5`).
   Exporta la nube binaria a
   `data/processed/palacio/nube_palacio_optimizada.ply` y guarda las figuras en
   `report/report_images/`. Si se cambia de objeto, adaptar también la ruta de
   entrada y la caja de recorte del notebook 03.
4. **Medición métrica**
   La última celda del notebook 03 calcula las dimensiones AABB y OBB en
   milímetros. Configurar `actual_height_mm` si se dispone de una medida física
   para calcular el error relativo. Las dimensiones actuales son provisionales:
   hay que validar el recorte antes de aceptarlas como medidas del objeto.

Después de recalibrar, ejecutar los notebooks 02 y 03 completos para mantener
coherentes los mapas, las disparidades, las poses y la nube final. Al preparar una
nueva entrega, incluir también la versión actualizada del `.ply` optimizado.

Las llamadas a `show_geometries()` abren el visor de Open3D y la celda continúa
al cerrar la ventana. Se pueden omitir las visualizaciones y las capturas si el
entorno no tiene soporte gráfico.

## Notas

- Los archivos de proceso intermedio pesados (p. ej. la nube de toda la escena
  sin optimizar) se excluyen del entregable.
