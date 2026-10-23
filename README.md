# TP2 — Reconstrucción 3D por estéreo multivista

I308 Visión Artificial — Universidad de San Andrés
Grupo: Agustín Patruno Carranza · Catalina Hirsch · Mathias Kneipp

Pipeline de fotogrametría (estéreo multivista) para reconstruir la nube de
puntos 3D de un objeto y estimar sus dimensiones métricas reales, usando
**OpenCV** y **Open3D**.

## Entregable final

- **Nube del Palacio optimizada (`.ply` binario):**
  [`data/processed/palacio/nube_palacio_optimizada.ply`](data/processed/palacio/nube_palacio_optimizada.ply),
  **incluida en este entregable** (4,2 MB, 155.401 puntos del objeto segmentado).
- **[Informe PDF](report/patruno_hirsch_kneipp-tp2-informe.pdf)** — incluido.
- **Dataset de imágenes propio (Google Drive):**
  [carpeta pública](https://drive.google.com/drive/folders/12rf3tCLWvMRrGbgWVxk7QTSbeLpmx9QJ?usp=sharing)
  con los pares de calibración y del Palacio. Por su peso (~87 MB) **no se incluye**
  en el entregable; ver la sección [Dataset](#dataset) para descargarlo y ubicarlo.

Los archivos de proceso intermedio pesados (la nube global sin optimizar, las
disparidades y la malla Poisson) se excluyen del entregable; sólo se incluye la
nube optimizada final.

## Estructura del repositorio

```
.
├── data/                  # resultados de calibración y nube final
│   ├── calibration_results/ # .pkl, matrices legibles y metadatos
│   └── processed/palacio/ # nube optimizada incluida; intermedios excluidos
│   # calibration/ y raw/palacio/ NO van en el entregable: se descargan
│   # del Drive y se ubican aquí (ver sección Dataset).
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

## Dataset

El dataset propio de imágenes se distribuye por **Google Drive** por su tamaño
(~87 MB) y **no está incluido** en este entregable:

- **[Descargar dataset (Google Drive)](https://drive.google.com/drive/folders/12rf3tCLWvMRrGbgWVxk7QTSbeLpmx9QJ?usp=sharing)**

La carpeta contiene dos subcarpetas. Para ejecutar los notebooks, descargalas y
ubicalas dentro del proyecto así:

- `calibration/` → `data/calibration/` (63 pares del tablero de calibración)
- `raw/palacio/` → `data/raw/palacio/` (72 pares estéreo del objeto)

Una vez colocadas, la estructura de `data/` queda completa y los notebooks
corren sin cambios.

## Uso

> **Antes de ejecutar:** descargar el dataset del [Drive](#dataset) y ubicarlo en
> `data/calibration/` y `data/raw/palacio/` como se indica arriba.

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
guarda como `False`: no se dispone de una comprobación física documentada del
patrón. La escala debe validarse midiendo el lado de los cuadrados impresos del
checkerboard y del ChArUco, y contrastando una distancia conocida con la
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
   carga la nube global, aplica el recorte espacial (caja holgada
   `[90, 55, -140]`–`[270, 500, 0]` mm), submuestreo de 1,5 mm y
   eliminación estadística de outliers (`nb_neighbors=20`, `std_ratio=1.5`), y
   finalmente segmenta el objeto (máscara `Z < -12 mm` + mayor grupo DBSCAN).
   Exporta la nube del objeto a
   `data/processed/palacio/nube_palacio_optimizada.ply` y guarda las figuras en
   `report/report_images/`. Si se cambia de objeto, adaptar también la ruta de
   entrada y la caja de recorte del notebook 03.
4. **Medición métrica**
   El **Paso 5** del notebook 03 mide el objeto ya aislado por su extensión robusta
   entre percentiles 0,5–99,5 (principal), con la AABB y la OBB como complemento.
   Las medidas físicas del Palacio, tomadas desde la base, son **38 cm de largo,
   12 cm de ancho y 11,5 cm de alto** (380 × 120 × 115 mm), cargadas como
   `actual_length_mm`, `actual_width_mm` y `actual_height_mm`. La caja de recorte se
   eligió holgada: la AABB del objeto deja holgura clara en los tres ejes
   (16,4/17,3 mm en X, 10,5/16,1 mm en Y, 6,5/12,0 mm en Z), por lo que la medición
   **no depende del recorte**. Las extensiones robustas actuales son
   **108,4 × 371,7 × 108,4 mm** (ancho × largo × alto): las tres **subestiman**
   levemente el objeto (largo **2,2 %**, alto **5,7 %**, ancho **9,7 %**), siendo el
   ancho —la dimensión más angosta— la peor estimada; la AABB, en cambio, las
   **sobreestima** por el engrosamiento lateral (ruido de disparidad y errores de
   pose), de modo que el ancho real (120 mm) queda acotado entre la robusta (108,4 mm)
   y la AABB (146,3 mm). La escala sigue pendiente de validación
   física de los patrones. Los pasos 6 y 7 muestran las poses y realizan
   la exploración opcional de superficie con Poisson.

Después de recalibrar, ejecutar los notebooks 02 y 03 completos para mantener
coherentes los mapas, las disparidades, las poses y la nube final. Al preparar una
nueva entrega, incluir también la versión actualizada del `.ply` optimizado.

Las llamadas a `show_geometries()` abren el visor de Open3D y la celda continúa
al cerrar la ventana. Se pueden omitir las visualizaciones y las capturas si el
entorno no tiene soporte gráfico.

## Notas

- Los archivos de proceso intermedio pesados (p. ej. la nube de toda la escena
  sin optimizar) se excluyen del entregable.
