# counting-mlops

Operaciones simples de MLOps para estrategias de conteo de objetos en video.
Cada experimento se define en un archivo YAML y se ejecuta con un único comando,
en `localhost` o en Modal.

## Operaciones soportadas

- `count`: corre la estrategia de conteo sobre cada video del dataset y escribe
  el conteo por video.
- `evaluate`: igual que `count` y, además, calcula el ground truth de cada video
  y el error porcentual contra el conteo del modelo.
- `plot`: genera una imagen con una serie por trial de un run (error `%` por
  video para `evaluate`, conteo por video para `count`).

El conteo se resuelve con una de dos estrategias, elegida en el campo `strategy`
del YAML:

- `line_crossing`: corre YOLO con tracking sobre el video y cuenta los objetos
  que cruzan una línea virtual.
- `embeddings`: corre YOLO detect por frame, filtra por una región, recorta las
  detecciones, las codifica con un modelo de embeddings y cuenta identidades
  únicas por matching frame a frame.

## Comandos

`CONFIG` y `RUN` son obligatorios: no hay valores por defecto, el experimento
siempre se elige de forma explícita. Cada bloque muestra la versión con Makefile
(recomendada) y su equivalente directo.

`make count` y `make evaluate` ejecutan el mismo runner; la operación la define
el campo `type` del YAML.

```bash
# instala dependencias con uv
make install
uv sync

# ejecuta un experimento (count o evaluate, segun el type del YAML)
make count    CONFIG=experiments/example.yaml
make evaluate CONFIG=experiments/example.yaml
uv run python -m ops run experiments/example.yaml

# genera y abre la imagen por trial
make plot RUN=runs/example/20260520_201945
uv run python -m ops plot runs/example/20260520_201945

# borra el directorio runs/
make clean

# lista los targets disponibles
make help
```

Los resultados se guardan en `runs/<name>/<YYYYMMDD_HHMMSS>/` con la copia del
YAML y los conteos o métricas. Si el `runtime` fue `modal`, cada trial se
descarga al terminar su contenedor, sin esperar al lote completo.

## Esquema del YAML

El archivo `experiments/example.yaml` controla los parametros del experimento.
Los cuales son:

- name: identificador del experimento, define la subcarpeta `runs/<name>/`
- type: operación a ejecutar (`count` | `evaluate`)
- strategy: estrategia de conteo (`line_crossing` | `embeddings`)
- runtime: `localhost` o `modal`
- device: `gpu` o `cpu`
- render: opcional, `true` escribe un `.mp4` anotado por video (solo
  `line_crossing`)
- dataset: ruta al `manifest.yaml` del dataset
- model: pesos del detector y, para `embeddings`, del encoder
- params: parámetros específicos de la estrategia

Cada parametro de `params` puede aceptar uno o varios valores separados por
comas. Cuando posee un solo valor el experimento se ejecuta bajo esa
combinacion; cuando son mas de uno se expande el producto cartesiano de todas
las combinaciones y ejecuta un trial por cada una. `region` queda fuera del
sweep porque su propio valor ya es una lista `x1,y1,x2,y2`.

Cada trial se guarda en un subdirectorio `<idx>_<key>-<val>_...` dentro del run,
nombrando los params que varían. Si solo hay un trial, los outputs van directo a
la raíz del run sin subdirectorio.

Las keys sweepables dependen de la estrategia: `count_mode`, `threshold`,
`direction`, `conf_threshold` y `crop` para `line_crossing`; `conf_threshold`,
`crop_padding`, `similarity_threshold`, `max_staleness` y `min_track_length`
para `embeddings`.

```yaml
name: example # identificador del experimento, define la subcarpeta runs/<name>/
type:
  evaluate # count | evaluate
  #   count     -> cuenta sobre cada video, escribe counts.json
  #   evaluate  -> cuenta y compara contra ground truth, escribe metrics.json
strategy:
  line_crossing # line_crossing | embeddings
  #   line_crossing -> YOLO con tracking + conteo por cruce de linea
  #   embeddings    -> YOLO detect + recorte + matching por embeddings
runtime: localhost # localhost | modal  (modal requiere el token de Modal configurado)
device: gpu # gpu | cpu
render: false # true escribe un .mp4 anotado por video en renders/ (solo line_crossing)
dataset: data/blueberry-counting/manifest.yaml # ruta al manifest.yaml del dataset

model:
  detector_weights: models/yolov8n_blueberry.pt # pesos YOLO (.pt)
  # encoder_weights: models/embeddings_v1.pt     # requerido si strategy=embeddings

# params para strategy: line_crossing
params:
  count_mode: vertical # vertical (eje y) | horizontal (eje x)
  threshold: 0.5 # posicion normalizada [0,1] de la linea de conteo
  direction: top2down # top2down | down2top (vertical) -- left2right | right2left (horizontal)
  conf_threshold: 0.25 # confianza minima de las detecciones YOLO
  crop: none # none | center_square (recorta un cuadrado central antes de pasar el frame a YOLO)


# Para strategy: embeddings, reemplaza el bloque params por:
# params:
#   region: 0.0,0.45,1.0,0.55  # x1,y1,x2,y2 normalizado de la region de conteo
#   conf_threshold: 0.25       # confianza minima de las detecciones YOLO
#   crop_padding: 0.5          # padding fraccional del recorte
#   similarity_threshold: 0.5  # corte de similitud coseno para el match
#   max_staleness: 2           # frames que un track aguanta sin match
#   min_track_length: 2        # minimo de frames para contar una identidad
```

El dataset es un directorio bajo `data/` con los videos y sus labels por frame.
Los labels son YOLO extendido (`class cx cy w h track_id`) en coordenadas
normalizadas `[0, 1]`. El `manifest.yaml` lista los `video_ids`:

```yaml
videos:
  - vid_001
  - vid_002
```

Cada estrategia evalúa contra su ground truth natural: `line_crossing` usa cruce
estricto de línea sobre los tracks etiquetados; `embeddings` usa las identidades
únicas cuyo centro cae dentro de la región.

## Resultados

La comparación line_crossing vs tiled_crossing y la ablation de `conf_threshold`
están en [RESULTS.md](RESULTS.md).

## Estructura

```
counting-mlops/
├── data/                          datasets de video
│   └── <dataset>/
│       ├── videos/<video_id>.mp4
│       ├── labels/<video_id>/frame_NNNNNN.txt  class cx cy w h track_id
│       └── manifest.yaml          lista de video_ids del dataset
├── experiments/                   archivos YAML, uno por experimento
│   └── example.yaml               plantilla documentada con todos los campos
├── models/                        pesos .pt (gitignored)
├── runs/                          outputs (gitignored)
│   └── <name>/<YYYYMMDD_HHMMSS>/
│       ├── config.yaml            copia del YAML usado
│       ├── error_by_trial.png     generado por `ops plot` (count_by_trial.png si type count)
│       └── <idx>_<key>-<val>_/    un subdirectorio por trial (si hay sweep)
│           ├── config.yaml        config resuelto del trial
│           ├── counts.json        conteo por video (type count)
│           ├── metrics.json       error por video y agregado (type evaluate)
│           └── renders/<video_id>.mp4   video anotado por video (si render: true)
├── ops/                           código de las operaciones
│   ├── __main__.py                entry: `python -m ops run|plot`
│   ├── config.py                  parser, validación y expansión de sweeps
│   ├── dataset.py                 carga del manifest y rutas del dataset
│   ├── groundtruth.py             ground truth desde labels con track_id
│   ├── count.py                   operación count
│   ├── evaluate.py                operación evaluate
│   ├── plot.py                    operación plot
│   ├── strategies/
│   │   ├── line_crossing.py       conteo por cruce de línea
│   │   └── embeddings.py          conteo por embeddings
│   └── runtimes/
│       ├── local.py               ejecuta en localhost
│       └── modal.py               ejecuta en Modal
├── Makefile
├── pyproject.toml
└── README.md
```
