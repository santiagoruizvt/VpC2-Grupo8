# VpC2-Grupo8

Este repositorio contiene todo lo relacionado con el trabajo práctico correspondiente a la asignatura "Visión por Computadora 2" de la CEIA.

## Detección de espacios libres en estacionamientos

Sistema que, a partir de la imagen completa de un estacionamiento, detecta las plazas **libres** y **ocupadas**
con un modelo de detección de objetos y calcula el **porcentaje de ocupación**.

**Alcance del TP**

- Comparar al menos dos arquitecturas de Deep Learning para detección (por ejemplo, YOLO vs Faster R-CNN / RT-DETR).
- Métricas: precision, recall, F1, mAP@0.5, mAP@0.5:0.95 y tiempo de inferencia.
- Cálculo automático del porcentaje de ocupación.
- Demo final: inferencia sobre un video corto de un estacionamiento, mostrando cómo se actualiza la disponibilidad.

## Dataset

[PKLot](https://www.kaggle.com/datasets/ammarnassanalhajali/pklot-dataset) (export de Roboflow *PKLot v2 640*, licencia CC BY 4.0):
12.416 imágenes de 640×640 de 3 cámaras fijas (PUCPR, UFPR04, UFPR05), anotadas en COCO con 2 clases:
`space-empty` y `space-occupied`.

> Almeida, P., Oliveira, L. S., Silva Jr, E., Britto Jr, A., Koerich, A., *PKLot – A robust dataset for parking lot classification*,
> Expert Systems with Applications, 42(11):4937-4949, 2015.

El dataset **no se versiona**. Descargarlo de Kaggle y descomprimirlo en `archive/` en la raíz del repo:

```
archive/
├── train/   (*.jpg + _annotations.coco.json)
├── valid/
└── test/
```

## Entorno

Se usa [uv](https://docs.astral.sh/uv/) con Python 3.12 (definido en `.python-version`).

```bash
uv sync                 # crea .venv e instala dependencias (incluye el grupo dev: jupyter, ipykernel)
uv run jupyter lab      # o abrir los notebooks en VS Code eligiendo el kernel de .venv
```

Si `uv sync` falla con `invalid peer certificate: UnknownIssuer` (proxy/antivirus que intercepta TLS), usar
los certificados del sistema: `uv sync --native-tls` (o la variable de entorno `UV_NATIVE_TLS=1`).

Para agregar dependencias: `uv add <paquete>` (y commitear `pyproject.toml` + `uv.lock`).

## Estructura

```
notebooks/
└── 01_exploracion_pklot.ipynb   # EDA del dataset
```

## Hallazgos principales de la exploración

- Clases balanceadas (~53% libres / 47% ocupadas). La categoría `spaces` (id 0) no se usa.
- La cámara no viene en los metadatos, pero se recupera por la cantidad de plazas anotadas: PUCPR = 100, UFPR05 = 40, UFPR04 = 28.
- 274 imágenes sin anotaciones (UFPR04 vacío, error de etiquetado) → descartar.
- Sólo un subconjunto de las plazas visibles está anotado → habrá que definir una ROI en la demo.
- Fuerte *leakage* temporal entre splits (frames a ~5 min de distancia) → evaluar también con split por día / por cámara.
