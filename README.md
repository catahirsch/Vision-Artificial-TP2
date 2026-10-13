# TP2 — Reconstrucción 3D por estéreo multivista

I308 Visión Artificial — Universidad de San Andrés
Grupo: Agustín Patruno Carranza · Catalina Hirsch · Mathias Kneipp

Pipeline de fotogrametría (estéreo multivista) para reconstruir la nube de
puntos 3D de un objeto y estimar sus dimensiones métricas reales, usando
**OpenCV** y **Open3D**.

## Entregable final

- **Nube de puntos final optimizada (`.ply` binario):** `<LINK A GOOGLE DRIVE>`
  > No se incluye en el repositorio por su tamaño (orden de decenas de MB).
- **Dataset propio (opcional):** `<LINK A GOOGLE DRIVE>`

## Estructura del repositorio

```
.
├── data/                  # fotografías y resultados de calibración
│   ├── calibration/       # 63 pares del tablero de calibración
│   ├── calibration_results/ # .pkl, matrices legibles y metadatos
│   ├── raw/               # palacio (72 pares) y peluche unificado (70 pares)
│   └── examples/          # imágenes de demostración de la cátedra
├── code_examples/         # código de referencia de la cátedra
├── src/                   # código fuente del pipeline (ver abajo)
├── report/                # informe (LaTeX + PDF)
└── README.md
```

## Requisitos

- Python 3.10+
- Dependencias (ver `requirements.txt`):

```
opencv-contrib-python==<version>
open3d==<version>
numpy==<version>
```

Instalación:

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

Los resultados de calibración existentes están en
[`data/calibration_results`](data/calibration_results), con los parámetros del
tablero registrados en [`metadata.yaml`](data/calibration_results/metadata.yaml).
El comando recuperado del historial de zsh indica un checkerboard de **9 × 6
esquinas internas** (10 × 7 cuadrados), con cuadrados de **28 mm de lado**.
El patrón mide **280 × 196 mm**, sin incluir márgenes. Antes de reportar medidas
en milímetros, confirmar que ese comando generó los resultados guardados y
validar la escala.

> Completar a medida que se implementa cada etapa.

1. **Calibración y rectificación estéreo**
   ```bash
   # <comando>
   ```
2. **Disparidad + nube local + pose global** (por cada par de `captures/`)
   ```bash
   # <comando>
   ```
3. **Fusión y optimización de la nube + export `.ply`**
   ```bash
   # <comando>
   ```
4. **Medición métrica**
   ```bash
   # <comando>
   ```

## Google Colab

`<LINK AL NOTEBOOK DE COLAB>`

## Notas

- Los archivos de proceso intermedio pesados (p. ej. la nube de toda la escena
  sin optimizar) se excluyen del entregable.
