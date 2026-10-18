# FALTANTES — TP2 Reconstrucción 3D

Lista de lo que falta **corregir**, **agregar al informe** y **agregar/ajustar en los
notebooks**, anclada punto por punto a la consigna
(`I308 Vision Artificial - TP2 - Reconstrucción 3D.pdf`).

Prioridad: 🔴 bloqueante / lo pide explícitamente la consigna · 🟡 recomendado ·
🟢 limpieza / pulido.

---

## 1. Entregables (consigna §6 "Informe, entregables y grupos de trabajo")

La consigna exige, como entregable:

- [x] **Informe** (≤ 8 páginas). Actual: 7 páginas. ✅
- [x] **Código fuente + `README.md` con instrucciones y versiones de dependencias.** ✅
- [x] **Estimación del tamaño del objeto en milímetros.** ✅ (AABB 90.0 × 380.0 × 129.8 mm)
- [ ] 🔴 **Archivo `.ply` optimizado (mandatorio), descargable de forma directa del
  campus o por link en la nube, "sin tener que bajar todo el proyecto".**
  Hoy el `.ply` (5,47 MB) solo está versionado en el repo y el README apunta a
  "GitHub → Download raw". **Hay que subir ese `.ply` al campus o a un link directo
  (Drive)** y ponerlo en la entrega, porque la consigna pide descarga directa sin bajar
  todo el proyecto.
- [ ] 🟡 **Peso del `.ply`.** La consigna dice "en el orden de **decenas** de Megabytes".
  El actual pesa 5,47 MB (por debajo). No es error, pero si se quiere entrar en el rango
  sugerido, bajar el `voxel_size` de optimización (p. ej. 1,5 mm → 1,0 mm) en el
  notebook 03.
- [ ] 🟡 **Dataset propio.** La consigna pide incluir el/los dataset(s) propios (link si
  pesan). Está `data/raw/palacio` y `data/raw/peluche`, pero **el informe nunca menciona
  el "peluche"** y existe un directorio vacío `capturespeluche/`. Decidir: o se reporta
  el peluche como captura adicional, o se lo excluye de la entrega para no generar dudas.
- [ ] 🟢 **Excluir intermedios pesados** (consigna §6: "Excluir del entregable archivos
  de proceso intermedio pesados, como la nube de puntos de toda la escena sin
  optimizar"). En `git` está bien excluido vía `.gitignore`, pero en disco queda
  `data/processed/palacio/nube_global_palacio.ply` (**2,35 GB**). Si se entrega
  comprimiendo la carpeta, **no incluirlo**.

---

## 2. Informe — contenido que la consigna pide y falta

### 2.1 🔴 Visualización de la nube local (consigna §6, Resultados → Disparidad)
> "Disparidad: mapa de disparidad de al menos un par representativo, **visualización de
> nube local**."

- El informe tiene el **mapa de disparidad** (Figura 3) ✅, pero **falta la figura de la
  nube de puntos 3D local de un par** (la "Computar Nube local" de §4.2).
- El notebook 02 ya la genera en pantalla (celda 6, `show_geometries([point_cloud_0])`)
  pero no la guarda como imagen → ver tarea 3.1.

### 2.2 🔴 Comparación con el tamaño real / error relativo (consigna §6, Resultados → Medición métrica)
> "tamaño estimado del objeto en milímetros, indicando método de medición utilizado, y
> **comparación con el tamaño real si se conoce (reportando error relativo)**."

- Como el objeto es propio y no se tomó una medida física de referencia, **hay que
  decirlo explícitamente en Resultados → Medición métrica**: "no se dispone de medida de
  referencia, por lo que no se reporta error relativo". Así la omisión queda justificada
  y no parece olvido. (Hoy solo se insinúa en la Discusión.)
- Alternativa mejor: medir con regla el alto real del objeto y completar
  `actual_height_mm` en el notebook 03 (celda 18) para reportar el error relativo real.

### 2.3 🟡 Descripción del dataset propio (consigna §3 "Recolección de imágenes propias")
- El informe nunca presenta los datos de forma ordenada: **qué objeto** se reconstruye
  (maqueta del "Palacio"), **con qué cámara** (la ELP estéreo de la cátedra), **cuántos
  pares de captura** (72) y de **calibración** (63), y que el **marco de referencia** es
  un tablero ChArUco visible en todas las tomas (consigna §3 pide objeto de referencia
  visible en todas las capturas).
- Falta un párrafo breve al inicio de **Metodología** (o de Resultados) con esa
  descripción. Hoy esos datos aparecen sueltos y tarde (p. ej. "72 pares" recién en la
  subsección de Pose).

### 2.4 🟡 Parámetros usados en el código y no mencionados (consigna §6, Metodología → Disparidad: "parámetros ajustados")
- **Filtro WLS:** se menciona el filtro pero no sus parámetros (λ = 8000, σ = 1.5).
- **`alpha = 0` de la rectificación** (`cv.stereoRectify`): recorta los bordes inválidos
  y define el FOV útil; no se menciona.
- **`minDisparity = 0`:** se dice "disparidad superior al mínimo configurado" sin dar el
  valor.
- **Dimensiones del ChArUco** (9 × 13 cuadrados, 42 mm, marcadores 31,5 mm,
  `DICT_5X5_100`): aparecen dispersas; conviene darlas donde se introduce el tablero, en
  la subsección de Pose.

### 2.5 🟡 Decisión sobre la malla de Poisson (consigna §4.3, opcional)
- Se agregó la **Figura 7 (malla Poisson)** al informe, pero **no existe código** que la
  genere (ni en notebooks ni en `src/`). La imagen `palacio_malla_poisson.png` está
  suelta y no es reproducible.
- Decidir una de dos:
  - **Agregar** el código al notebook 03 (`estimate_normals` +
    `create_from_point_cloud_poisson`) para que la figura sea reproducible, o
  - **Quitar** la Figura 7 del informe.

### 2.6 🟢 Pulido menor
- Pie de la Figura 6 (`poses_palacio.png`) dice "Nube global antes del recorte", pero la
  imagen también incluye el origen del mundo y los ejes/frustums de las cámaras (nb02,
  celda 10). Ajustar el pie para reflejar que también muestra las poses.
- Mencionar en Metodología → Calibración el **filtrado iterativo por umbral τ = 3 px**
  (de 49 → 13 pares): es una poda agresiva y conviene justificar por qué se descartan
  tantos pares (hoy solo se describe el resultado, no el criterio, en Resultados).

---

## 3. Notebooks — ajustes

### 3.1 🔴 Guardar la captura de la nube local (para la tarea 2.1)
- En `notebooks/02_reconstruction.ipynb`, celda 6, agregar un
  `save_open3d_screenshot([point_cloud_0], ...)` que escriba la imagen en
  `report/report_images/` (igual que se hace con las nubes de optimización en el
  notebook 03), y luego insertar esa figura en el informe.

### 3.2 🟡 Poisson reproducible (solo si se mantiene la Figura 7 — tarea 2.5)
- En `notebooks/03_optimization_measurement.ipynb`, tras la nube optimizada, agregar:
  `clean_point_cloud.estimate_normals(...)` → `create_from_point_cloud_poisson(...)` y
  guardar la captura en `report/report_images/palacio_malla_poisson.png`.

### 3.3 🟡 Medida real para el error relativo (para la tarea 2.2)
- En `notebooks/03_optimization_measurement.ipynb`, celda 18, asignar
  `actual_height_mm` con la medida física del objeto (si se la toma) para que se calcule
  e imprima el error relativo.

---

## 4. Limpieza del repositorio 🟢

- [ ] Borrar el directorio vacío `capturespeluche/`.
- [ ] Borrar `notebooks/RenderOption_2026-10-16-17-47-09.json` (generado por el visor de
  Open3D).
- [ ] Al preparar la entrega, asegurarse de **no** incluir
  `data/processed/palacio/nube_global_palacio.ply` (2,35 GB) ni las disparidades
  intermedias (consigna §6).

---

## Estado de lo que YA cumple la consigna ✅

- §4.1 Calibración y rectificación: implementación manual con OpenCV
  (`findChessboardCorners`, `calibrateCamera`, `stereoCalibrate`, `stereoRectify`,
  `initUndistortRectifyMap`, `remap`), parámetros intrínsecos/distorsión/extrínsecos,
  error de reproyección RMS, y figura original vs. rectificado.
- §4.2 Disparidad (SGBM + WLS), nube local con `reprojectImageTo3D` y `Q`, pose global
  con `solvePnP` + `Rodrigues`, transformación local → mundo con `T_cw = [Rᵀ | −Rᵀt]`.
- §4.3 Fusión y optimización en Open3D: recorte (bounding box) → voxel → outliers, con
  conteo de puntos por etapa y export `.ply` **binario**.
- §5 Medición métrica: dimensiones en mm con AABB y OBB, y justificación de por qué el
  resultado es métrico (sin ambigüedad de escala).
- Marco teórico (pinhole y distorsión, calibración estéreo, geometría epipolar y
  rectificación, disparidad → profundidad), Metodología y Resultados y Discusión.
- Abstract y Discusión completos; informe en 7 páginas (≤ 8).
