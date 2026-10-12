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
├── datasets/              # datasets provistos por la cátedra (calib/ + captures/)
│   ├── stereo_budha_board    # 6 capturas, checkerboard
│   └── stereo_budha_charuco  # 21 capturas en órbita 360°, ChArUco
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
