# Resultados

Comparación de dos estrategias de conteo por cruce de línea sobre el dataset
blueberry-counting: `line_crossing` (baseline, YOLO + tracking nativo) y
`tiled_crossing` (la franja central del frame partida en 2 tiles, con tracking y
conteo independientes por tile).

Las métricas son sobre los 5 videos del dataset; `mae` es el error absoluto
medio.

## line_crossing vs tiled_crossing por detector

`conf_threshold` 0.15 fijo, mismos params, 20 detectores YOLO entrenados en
arándanos. Experimentos: `experiments/blueberry_line_detectors.yaml`,
`experiments/blueberry_tiled_detectors.yaml`,
`experiments/blueberry_line_yolov11.yaml`,
`experiments/blueberry_tiled_yolov11.yaml`,
`experiments/blueberry_line_yolo26.yaml` y
`experiments/blueberry_tiled_yolo26.yaml` (`detector_weights` sweepable, un
trial por detector).

| detector | line_crossing mae | tiled_crossing mae |
| -------- | ----------------- | ------------------ |
| yolov8n  | 76.6%             | 57.9%              |
| yolov8s  | 67.8%             | 33.2%              |
| yolov8m  | 68.3%             | 53.4%              |
| yolov8l  | 71.4%             | 37.2%              |
| yolov9t  | 60.5%             | 35.9%              |
| yolov9s  | 61.9%             | 22.8%              |
| yolov9m  | 69.7%             | 62.8%              |
| yolov9c  | 74.9%             | 41.1%              |
| yolov10n | 76.3%             | 71.6%              |
| yolov10s | 66.6%             | 47.7%              |
| yolov10m | 61.3%             | 29.2%              |
| yolov10l | 67.6%             | 45.5%              |
| yolov11n | 57.5%             | 28.0%              |
| yolov11s | 67.7%             | 25.5%              |
| yolov11m | 74.5%             | 36.8%              |
| yolov11l | 78.0%             | 47.8%              |
| yolo26n  | 74.7%             | 62.1%              |
| yolo26s  | 67.1%             | 47.9%              |
| yolo26m  | 67.1%             | 72.4%              |
| yolo26l  | 65.9%             | 33.1%              |

`tiled_crossing` mejora a `line_crossing` en 19 de los 20 detectores (única
excepción: `yolo26m`, donde tiled queda 5.3 puntos peor). La mejor
combinación es `tiled_crossing` + `yolov9s`, con `mae` 22.8% (bias -13.8%).
Ambas estrategias subcuentan siempre: el techo lo fija el detector, que en
escena densa solo detecta cerca de la mitad de los arándanos.

## Sensibilidad a conf_threshold (tiled_crossing + yolov9s)

Experimento: `experiments/blueberry_tiled_yolov9s_conf.yaml`.

| conf_threshold | mae   | bias   |
| -------------- | ----- | ------ |
| 0.05           | 22.8% | -13.8% |
| 0.10           | 22.8% | -13.8% |
| 0.15           | 22.8% | -13.8% |
| 0.20           | 22.8% | -13.8% |
| 0.30           | 23.2% | -15.4% |
| 0.40           | 24.3% | -20.3% |

El resultado es estable para `conf_threshold` entre 0.05 y 0.20.

## ReID (tiled_crossing + yolov9s)

BoT-SORT por defecto con `with_reid` activado (`model: auto`, features del
backbone del detector, sin modelo de ReID dedicado ni tuning), `conf_threshold`
0.15.

| configuración | mae   |
| ------------- | ----- |
| sin ReID      | 22.8% |
| con ReID      | 21.6% |

Activar ReID no cambia el resultado de forma significativa: la apariencia aporta
poca señal porque los arándanos son diminutos y casi idénticos entre sí, así que
la asociación del tracker no es el cuello de botella.
