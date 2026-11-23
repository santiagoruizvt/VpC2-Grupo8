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

Los notebooks 01 y 02 corren con este entorno. El 03 necesita además `ultralytics` y una GPU, por lo que está pensado para Google Colab
(ver más abajo).

## Estructura

```
notebooks/
├── 01_exploracion_pklot.ipynb      # EDA del dataset
├── 02_preparacion.ipynb            # limpieza, conversión COCO a YOLO y estrategias de split
└── 03_entrenamiento_yolo.ipynb     # entrenamiento y evaluación de YOLO (pensado para Colab)
```

Lo que generan los notebooks **no se versiona** (debe estar en el `.gitignore`): `archive/`, `datasets/`, `runs/`, `*.pt` y `*.zip`.

## Cómo reproducir el flujo

### 1. Preparar los datos (`02_preparacion.ipynb`)

1. Dejar el dataset en `archive/` (ver sección Dataset) y abrir el notebook con el kernel de `.venv`.
2. Ejecutarlo completo (no necesita GPU). Genera `datasets/pklot_yolo/` con:

   | Archivo | Contenido |
   |---|---|
   | `images/all/`, `labels/all/` | imágenes limpias y etiquetas YOLO |
   | `annotations_clean.coco.json` | COCO limpio (ids 1 = `space-empty`, 2 = `space-occupied`) para torchvision |
   | `metadata.csv` | cámara, fecha, hora, ocupación y split de cada imagen en cada estrategia |
   | `<estrategia>_{train,valid,test}.txt` y `data_<estrategia>.yaml` | listas de archivos y configuración para Ultralytics |

3. Estrategias de split disponibles:

   | Estrategia | Qué mide |
   |---|---|
   | `oficial` | split de Roboflow (aleatorio por imagen, con *leakage* temporal; sirve para comparar con otros trabajos) |
   | `por_dia` | cada día completo cae en un único split: generalización a días nuevos |
   | `por_camara_PUCPR`, `por_camara_UFPR04`, `por_camara_UFPR05` | test = una cámara completa nunca vista: generalización a un estacionamiento nuevo |

4. Para entrenar en Colab, comprimir la carpeta de salida de modo que **dentro del zip quede `pklot_yolo/` en la raíz**:

   ```bash
   cd datasets && zip -qr pklot_yolo.zip pklot_yolo -x "*.DS_Store" "*.cache"
   ```

   En Windows se puede usar 7-Zip con la misma condición. Para que los resultados sean comparables entre integrantes,
   **todos deben usar el mismo zip**.

### 2. Entrenar y evaluar (`03_entrenamiento_yolo.ipynb`)

**Opción A: Google Colab (recomendada)**

1. Subir `pklot_yolo.zip` a Google Drive en `Mi unidad/vpc2/pklot_yolo.zip` (la carpeta `vpc2` se crea a mano).
   Si el zip se comparte desde la unidad de otra persona, agregar un acceso directo a la propia unidad o cambiar `DRIVE_BASE` en el notebook.
2. Abrir el notebook en Colab: *Archivo, Abrir cuaderno, pestaña GitHub* (si el repo es privado, Colab pide autorización), o subir el `.ipynb` a mano.
3. *Entorno de ejecución, Cambiar tipo de entorno de ejecución, GPU (T4)*.
4. En la celda de configuración elegir `STRATEGY` y `MODEL` y ejecutar de arriba hacia abajo. Colab pide autorizar el acceso a Drive.
   El notebook descomprime el zip en el disco local de Colab (no se entrena leyendo desde Drive).

**Opción B: máquina local con GPU NVIDIA**

Instalar `ultralytics` con `uv add ultralytics` (para PyTorch con CUDA ver el comentario de `pyproject.toml`) y ejecutar el notebook:
detecta que no está en Colab y usa `datasets/pklot_yolo/` y `runs/` dentro del repo.

**Parámetros** (primera celda del notebook)

| Parámetro | Valor actual | Para qué sirve |
|---|---|---|
| `STRATEGY` | `por_dia` | estrategia de split (ver tabla de arriba) |
| `MODEL` | `yolo11m.pt` | pesos preentrenados en COCO; también `yolo11s.pt`, `rtdetr-l.pt`, etc. |
| `EPOCHS` / `PATIENCE` | 30 / 10 | máximo de épocas y *early stopping* sobre el mAP de validación |
| `FRACTION` | 1.0 | fracción del train a usar; bajarla sólo para pruebas rápidas |

Para comparar modelos o estrategias entre sí conviene **no cambiar** `EPOCHS`, `PATIENCE`, `IMGSZ` ni `BATCH`.

**Resultados** (en Drive, `Mi unidad/vpc2/runs/`)

- `<modelo>_<estrategia>/weights/best.pt` y `last.pt`, curvas y `results.csv`.
- `metricas_<modelo>_<estrategia>.csv`: precision, recall, F1, mAP@0.5, mAP@0.5:0.95 y milisegundos de inferencia, global, por clase y por cámara.

**Si Colab se corta durante el entrenamiento**

1. Reconectar y ejecutar en orden: configuración, sección 1 (monta Drive y descomprime el zip) y sección 2 (define `meta`).
2. **No ejecutar la celda de entrenamiento original**: reentrena desde cero y pisa los resultados guardados en Drive.
3. Ejecutar la celda de reanudación (`resume=True`) y luego continuar con las secciones 4 y 5.

## Hallazgos principales de la exploración

- Clases balanceadas (~53% libres / 47% ocupadas). La categoría `spaces` (id 0) no se usa.
- La cámara no viene en los metadatos, pero se recupera por la cantidad de plazas anotadas: PUCPR = 100, UFPR05 = 40, UFPR04 = 28.
- 274 imágenes sin anotaciones (UFPR04 vacío, error de etiquetado) → descartar.
- Sólo un subconjunto de las plazas visibles está anotado → habrá que definir una ROI en la demo.
- Fuerte *leakage* temporal entre splits (frames a ~5 min de distancia) → evaluar también con split por día / por cámara.

## Estado y próximos pasos

- [x] EDA (01)
- [x] Preparación de datos y estrategias de split (02)
- [x] Baseline YOLO11m con `por_dia` (03)
- [ ] YOLO11m con `oficial` y con `por_camara_*`
- [ ] Segunda arquitectura (RT-DETR o Faster R-CNN) con la misma configuración
- [ ] Comparativa de métricas y tiempos de inferencia
- [ ] Porcentaje de ocupación por frame y definición de la ROI para la demo
- [ ] Demo sobre video con tracking
