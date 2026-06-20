# Post-procesado: intervalos de confianza

Las cifras puntuales de [RESULTS.md](RESULTS.md) (mae por detector y por
estrategia) ocultan que la evaluación se hace sobre `n=5` videos, una muestra
chica. Para cuantificar la incertidumbre asociada se aplica bootstrap no
paramétrico sobre `per_video.error_pct` de cada `metrics.json`.

## Método

- Resampleo con reemplazo de los 5 errores por video, `n_iters = 10000`.
- En cada iteración se calcula el `mae` (promedio de `|error_pct|`) y el `bias`
  (promedio con signo).
- El IC95% son los percentiles 2.5 y 97.5 de la distribución resultante.
- Semilla fija (`SEED = 0`) para que el resultado sea reproducible.

Implementación: `ops/bootstrap.py`. Subcomando:

```bash
uv run python -m ops bootstrap runs/<name>/<timestamp>
```

El subcomando recorre todos los `metrics.json` bajo el directorio, escribe un
bloque `"bootstrap": {...}` dentro de cada uno y muestra una tabla con el mae,
IC95% del mae, bias e IC95% del bias.

## Resultados por detector

`conf_threshold` 0.15 fijo, mismos params que en [RESULTS.md](RESULTS.md).

### line_crossing

| detector  | mae    | IC95% mae        | bias    | IC95% bias        |
| --------- | ------ | ---------------- | ------- | ----------------- |
| yolov8n   | 76.6%  | [69.6, 81.2]     | -76.6%  | [-81.2, -69.6]    |
| yolov8s   | 67.8%  | [56.0, 75.2]     | -67.8%  | [-75.2, -56.0]    |
| yolov8m   | 68.3%  | [58.0, 74.8]     | -68.3%  | [-74.8, -58.0]    |
| yolov8l   | 71.4%  | [60.8, 78.6]     | -71.4%  | [-78.6, -60.8]    |
| yolov9t   | 60.5%  | [51.9, 66.6]     | -60.5%  | [-66.6, -51.9]    |
| yolov9s   | 61.9%  | [50.4, 68.9]     | -61.9%  | [-68.9, -50.4]    |
| yolov9m   | 69.7%  | [61.4, 75.5]     | -69.7%  | [-75.5, -61.4]    |
| yolov9c   | 74.9%  | [64.9, 81.3]     | -74.9%  | [-81.3, -64.9]    |
| yolov10n  | 76.3%  | [67.9, 81.8]     | -76.3%  | [-81.8, -67.9]    |
| yolov10s  | 66.6%  | [55.2, 73.8]     | -66.6%  | [-73.8, -55.2]    |
| yolov10m  | 61.3%  | [49.8, 68.9]     | -61.3%  | [-68.9, -49.8]    |
| yolov10l  | 67.6%  | [55.9, 74.8]     | -67.6%  | [-74.8, -55.9]    |
| yolov11n  | 57.5%  | [48.0, 64.7]     | -57.5%  | [-64.7, -48.0]    |
| yolov11s  | 67.7%  | [57.6, 74.4]     | -67.7%  | [-74.4, -57.6]    |
| yolov11m  | 74.5%  | [62.9, 80.6]     | -74.5%  | [-80.6, -62.9]    |
| yolov11l  | 78.0%  | [68.0, 83.7]     | -78.0%  | [-83.7, -68.0]    |
| yolo26n   | 74.7%  | [66.9, 79.9]     | -74.7%  | [-79.9, -66.9]    |
| yolo26s   | 67.1%  | [57.9, 73.5]     | -67.1%  | [-73.5, -57.9]    |
| yolo26m   | 67.1%  | [59.6, 71.6]     | -67.1%  | [-71.6, -59.6]    |
| yolo26l   | 65.9%  | [56.5, 72.5]     | -65.9%  | [-72.5, -56.5]    |

### tiled_crossing

| detector  | mae    | IC95% mae        | bias    | IC95% bias        |
| --------- | ------ | ---------------- | ------- | ----------------- |
| yolov8n   | 57.9%  | [44.7, 71.2]     | -57.9%  | [-71.2, -44.7]    |
| yolov8s   | 33.2%  | [18.5, 47.6]     | -33.2%  | [-47.6, -18.5]    |
| yolov8m   | 53.4%  | [42.5, 64.3]     | -53.4%  | [-64.3, -42.5]    |
| yolov8l   | 37.2%  | [22.5, 51.3]     | -37.2%  | [-51.3, -22.5]    |
| yolov9t   | 35.9%  | [20.6, 50.2]     | -35.9%  | [-50.2, -20.6]    |
| yolov9s   | 22.8%  | [12.8, 32.7]     | -13.8%  | [-30.3, +4.4]     |
| yolov9m   | 62.8%  | [55.1, 70.5]     | -62.8%  | [-70.5, -55.1]    |
| yolov9c   | 41.1%  | [27.6, 54.5]     | -41.1%  | [-54.5, -27.6]    |
| yolov10n  | 71.6%  | [66.2, 76.7]     | -71.6%  | [-76.7, -66.2]    |
| yolov10s  | 47.7%  | [35.9, 59.2]     | -47.7%  | [-59.2, -35.9]    |
| yolov10m  | 29.2%  | [14.4, 44.1]     | -28.7%  | [-44.1, -13.1]    |
| yolov10l  | 45.5%  | [31.6, 58.0]     | -45.5%  | [-58.0, -31.6]    |
| yolov11n  | 28.0%  | [14.6, 41.4]     | -23.5%  | [-41.4, -5.4]     |
| yolov11s  | 25.5%  | [13.1, 37.5]     | -19.8%  | [-37.5, -0.8]     |
| yolov11m  | 36.8%  | [20.5, 52.8]     | -36.8%  | [-52.8, -20.5]    |
| yolov11l  | 47.8%  | [33.8, 60.7]     | -47.8%  | [-60.7, -33.8]    |
| yolo26n   | 62.1%  | [50.4, 73.7]     | -62.1%  | [-73.7, -50.4]    |
| yolo26s   | 47.9%  | [31.8, 60.1]     | -47.9%  | [-60.1, -31.8]    |
| yolo26m   | 72.4%  | [63.3, 81.4]     | -72.4%  | [-81.4, -63.3]    |
| yolo26l   | 33.1%  | [15.9, 49.4]     | -33.1%  | [-49.4, -15.9]    |

## Lectura

- La mejor combinación, `tiled_crossing` + `yolov9s`, queda en `mae 22.8%`,
  `IC95 [12.8, 32.7]`, `bias -13.8%`, `IC95 [-30.3, +4.4]`. El IC del bias
  cruza el cero, lo que indica que con n=5 no se puede afirmar con confianza
  que el método subcuente sistemáticamente, solo que tiende a hacerlo.
- Los IC son anchos (>10 puntos en casi todos los detectores) porque la
  muestra es pequeña. Varios detectores se solapan entre sí: `yolov9s`
  (22.8 [12.8, 32.7]) y `yolov11s` (25.5 [13.1, 37.5]) no son estadísticamente
  distinguibles con esta evaluación.
- Las anchuras hablan de la calidad de la evidencia, no del método: la única
  forma real de cerrarlos es etiquetar más videos.
