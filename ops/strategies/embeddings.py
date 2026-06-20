"""
Estrategia de conteo por embeddings.

Por cada frame del video: YOLO detecta, se filtran las detecciones cuyo centro
cae dentro de la región, se recortan con padding y se codifican con un modelo
de embeddings. El conteo de identidades únicas se hace con matching greedy
frame a frame por similitud coseno.

Port de las piezas de app/services/embeddings/ del backend de video-labeler:
  - EmbeddingNet  <- model.py
  - EmbeddingEncoder <- encoder.py
  - count_via_tracking <- tracker.py  (la ruta sin trace)

count_via_tracking asume movimiento de izquierda a derecha: invalida un match
cuando la detección está a la izquierda de la última posición conocida del
track. Es el mismo supuesto que el endpoint de producción.
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

INPUT_SIZE = 48
EMBED_DIM = 128

# Misma normalización que el entrenamiento del encoder.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# --------------------------------------------------------------------------
# Modelo de embeddings
# --------------------------------------------------------------------------
def _conv_block(in_ch: int, out_ch: int, stride: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class EmbeddingNet(nn.Module):
    """CNN pequeña para crops diminutos. Salida: embedding L2-normalizado."""

    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.backbone = nn.Sequential(
            _conv_block(3, 32, stride=2),     # 48 -> 24
            _conv_block(32, 64, stride=2),    # 24 -> 12
            _conv_block(64, 128, stride=2),   # 12 -> 6
            _conv_block(128, 128, stride=1),  # 6  -> 6
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(128, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h).flatten(1)
        h = self.proj(h)
        return F.normalize(h, dim=-1)


class EmbeddingEncoder:
    """Carga EmbeddingNet desde un checkpoint y codifica crops en vectores."""

    def __init__(self, weights_path: Path, device: str):
        self.weights_path = weights_path
        self.device = device
        self.model: EmbeddingNet | None = None

    def load(self) -> None:
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Pesos del encoder no encontrados: {self.weights_path}")
        model = EmbeddingNet()
        state = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model.eval().to(self.device)
        self.model = model

    def _preprocess(self, crops: list[np.ndarray]) -> torch.Tensor:
        """Redimensiona, normaliza y apila. Los crops son BGR uint8 (cv2)."""
        batch = np.empty((len(crops), 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        for i, crop in enumerate(crops):
            if crop.size == 0:
                batch[i] = 0
                continue
            resized = cv2.resize(crop, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - _MEAN) / _STD
            batch[i] = rgb.transpose(2, 0, 1)
        return torch.from_numpy(batch)

    @torch.inference_mode()
    def encode(self, crops: list[np.ndarray], batch_size: int = 256) -> np.ndarray:
        """Codifica una lista de crops BGR en embeddings (N, EMBED_DIM)."""
        if self.model is None:
            self.load()
        assert self.model is not None

        out: list[np.ndarray] = []
        for start in range(0, len(crops), batch_size):
            chunk = crops[start : start + batch_size]
            x = self._preprocess(chunk).to(self.device, non_blocking=True)
            out.append(self.model(x).cpu().numpy())
        if not out:
            return np.empty((0, EMBED_DIM), dtype=np.float32)
        return np.concatenate(out, axis=0)


# --------------------------------------------------------------------------
# Conteo por matching frame a frame
# --------------------------------------------------------------------------
def count_via_tracking(
    frames: list[list[np.ndarray]],
    encoder: EmbeddingEncoder,
    similarity_threshold: float,
    max_staleness: int,
    min_track_length: int,
    bboxes: list[list[dict]],
) -> tuple[int, list[list[int]]]:
    """Cuenta identidades únicas en una secuencia ordenada de frames.

    Args:
        frames: crops por frame (BGR uint8), en orden temporal.
        encoder: encoder de embeddings.
        similarity_threshold: corte de similitud coseno para considerar un
            match con un track activo. Más alto = más estricto.
        max_staleness: frames que un track activo aguanta sin match antes de
            descartarse.
        min_track_length: mínimo de frames en que debe aparecer un track para
            contar como identidad única. Filtra tracks espurios.
        bboxes: bboxes por frame (paralelo a frames), con x_center usado para
            invalidar matches contra el sentido del movimiento.

    Returns:
        (total, trace): total de identidades únicas que pasaron el filtro
        min_track_length, y trace[f] = lista con el track_idx de cada detección
        del frame f (paralela a frames y bboxes), para el render.
    """
    next_track_idx = 0
    # active: lista de (track_idx, embedding, staleness, last_x)
    active: list[tuple[int, np.ndarray, int, float]] = []
    track_appearances: dict[int, int] = {}
    trace: list[list[int]] = []

    for f_idx, crops in enumerate(frames):
        if not crops:
            if active:
                active = [
                    (idx, emb, stale + 1, lx)
                    for idx, emb, stale, lx in active
                    if stale + 1 <= max_staleness
                ]
            trace.append([])
            continue

        embeddings = encoder.encode(crops)
        det_x = [bboxes[f_idx][m]["x_center"] for m in range(len(crops))]
        det_to_track_idx: dict[int, int] = {}

        if not active:
            for m in range(len(crops)):
                det_to_track_idx[m] = next_track_idx
                active.append((next_track_idx, embeddings[m], 0, det_x[m]))
                next_track_idx += 1
        else:
            active_emb = np.stack([emb for _, emb, _, _ in active], axis=0)
            sims = embeddings @ active_emb.T  # (M, A)

            # Invalida matches donde la detección está a la izquierda de la
            # última posición del track (los objetos solo se mueven a la derecha).
            for m in range(sims.shape[0]):
                for a in range(sims.shape[1]):
                    if det_x[m] < active[a][3]:
                        sims[m, a] = -np.inf

            matched_actives: set[int] = set()
            det_to_active_pos: dict[int, int] = {}
            work = sims.copy()
            while True:
                idx = int(np.argmax(work))
                m, a = divmod(idx, work.shape[1])
                if work[m, a] < similarity_threshold:
                    break
                det_to_active_pos[m] = a
                matched_actives.add(a)
                work[m, :] = -np.inf
                work[:, a] = -np.inf

            new_active: list[tuple[int, np.ndarray, int, float]] = []

            # Tracks activos sin match: se arrastran si siguen dentro de staleness.
            for a_idx, (t_idx, emb, stale, lx) in enumerate(active):
                if a_idx in matched_actives:
                    continue
                if stale + 1 <= max_staleness:
                    new_active.append((t_idx, emb, stale + 1, lx))

            # Detecciones con match: heredan el track_idx y actualizan embedding.
            for m in range(len(crops)):
                if m in det_to_active_pos:
                    t_idx = active[det_to_active_pos[m]][0]
                    det_to_track_idx[m] = t_idx
                    new_active.append((t_idx, embeddings[m], 0, det_x[m]))

            # Detecciones sin match: identidades nuevas.
            for m in range(len(crops)):
                if m not in det_to_active_pos:
                    det_to_track_idx[m] = next_track_idx
                    new_active.append((next_track_idx, embeddings[m], 0, det_x[m]))
                    next_track_idx += 1

            active = new_active

        for t_idx in det_to_track_idx.values():
            track_appearances[t_idx] = track_appearances.get(t_idx, 0) + 1
        trace.append([det_to_track_idx[m] for m in range(len(crops))])

    total = sum(1 for count in track_appearances.values() if count >= min_track_length)
    return total, trace


# --------------------------------------------------------------------------
# Pipeline de conteo sobre un video
# --------------------------------------------------------------------------
def count_video(
    video_path: Path,
    detector_weights: Path,
    encoder_weights: Path | None,
    params: dict,
    device: str,
    render_path: Path | None = None,
) -> int:
    """Cuenta identidades únicas que pasan por una región en un video.

    Args:
        video_path: ruta al archivo .mp4.
        detector_weights: ruta a los pesos YOLO (.pt).
        encoder_weights: ruta a los pesos del encoder de embeddings (.pt).
        params: dict con region, conf_threshold, crop_padding,
            similarity_threshold, max_staleness, min_track_length.
        device: "gpu" | "cpu".
        render_path: si se da, escribe un .mp4 anotado con la región, las
            detecciones con su track id y el conteo acumulado.

    Returns:
        Número de identidades únicas contadas.
    """
    from ultralytics import YOLO

    if not video_path.exists():
        raise FileNotFoundError(f"Video no encontrado: {video_path}")
    if not detector_weights.exists():
        raise FileNotFoundError(f"Pesos del detector no encontrados: {detector_weights}")
    if encoder_weights is None or not encoder_weights.exists():
        raise FileNotFoundError(f"Pesos del encoder no encontrados: {encoder_weights}")

    model = YOLO(str(detector_weights))
    encoder = EmbeddingEncoder(encoder_weights, _torch_device(device))
    region = params["region"]
    conf = params["conf_threshold"]
    padding = params["crop_padding"]

    per_frame_crops: list[list[np.ndarray]] = []
    per_frame_bboxes: list[list[dict]] = []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(frame, conf=conf, device=_device(device), verbose=False)[0]
            crops, frame_bboxes = _crops_in_region(frame, result, region, padding)
            per_frame_crops.append(crops)
            per_frame_bboxes.append(frame_bboxes)
    finally:
        capture.release()

    total, trace = count_via_tracking(
        per_frame_crops,
        encoder,
        similarity_threshold=params["similarity_threshold"],
        max_staleness=params["max_staleness"],
        min_track_length=params["min_track_length"],
        bboxes=per_frame_bboxes,
    )

    if render_path is not None:
        _render_video(
            video_path,
            per_frame_bboxes,
            trace,
            region,
            params["min_track_length"],
            total,
            render_path,
        )

    return total


def _render_video(
    video_path: Path,
    per_frame_bboxes: list[list[dict]],
    trace: list[list[int]],
    region: tuple[float, float, float, float],
    min_track_length: int,
    total: int,
    render_path: Path,
) -> None:
    """Re-lee el video y escribe un .mp4 anotado: la región de conteo, la línea
    central, cada detección con su track id (verde si el track ya alcanzó
    min_track_length, amarillo si aún no) y el conteo acumulado."""
    render_path.parent.mkdir(parents=True, exist_ok=True)
    x1r, y1r, x2r, y2r = region
    line_x = (x1r + x2r) / 2

    capture = cv2.VideoCapture(str(video_path))
    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 0 else 30.0

    writer = None
    cumulative: Counter = Counter()
    running = 0
    try:
        for f_idx in range(len(trace)):
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]

            # Un track suma al conteo el frame en que llega a min_track_length.
            for t_idx in trace[f_idx]:
                cumulative[t_idx] += 1
                if cumulative[t_idx] == min_track_length:
                    running += 1

            cv2.rectangle(
                frame,
                (int(x1r * w), int(y1r * h)),
                (int(x2r * w), int(y2r * h)),
                (255, 150, 0),
                2,
            )
            cv2.line(frame, (int(line_x * w), 0), (int(line_x * w), h), (0, 0, 255), 2)

            for bbox, t_idx in zip(per_frame_bboxes[f_idx], trace[f_idx]):
                cx = int(bbox["x_center"] * w)
                cy = int(bbox["y_center"] * h)
                bw = bbox["width"] * w
                bh = bbox["height"] * h
                radius = int(round((1 / 3) * (bw**2 + bh**2) ** 0.5))
                counted = cumulative[t_idx] >= min_track_length
                color = (0, 255, 0) if counted else (0, 200, 200)
                cv2.circle(frame, (cx, cy), radius, color, -1 if counted else 1)
                cv2.putText(
                    frame,
                    str(t_idx),
                    (cx - radius, cy - radius - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            cv2.putText(
                frame,
                f"count: {running}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is None:
                writer = cv2.VideoWriter(
                    str(render_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer.write(frame)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
            print(f"[embeddings] video anotado escrito en {render_path}")

    assert running == total, f"[embeddings] conteo del render {running} != total {total}"


def _crops_in_region(
    frame: np.ndarray,
    result,
    region: tuple[float, float, float, float],
    padding: float,
) -> tuple[list[np.ndarray], list[dict]]:
    """Filtra las detecciones cuyo centro cae en la región y devuelve los
    crops (con padding) junto con sus bboxes normalizados."""
    x1r, y1r, x2r, y2r = region
    crops: list[np.ndarray] = []
    frame_bboxes: list[dict] = []

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return crops, frame_bboxes

    for cx, cy, bw, bh in boxes.xywhn.cpu().numpy():
        if not (x1r <= cx <= x2r and y1r <= cy <= y2r):
            continue
        crop = _crop_bbox(frame, float(cx), float(cy), float(bw), float(bh), padding)
        if crop is None:
            continue
        crops.append(crop)
        frame_bboxes.append(
            {"x_center": float(cx), "y_center": float(cy), "width": float(bw), "height": float(bh)}
        )
    return crops, frame_bboxes


def _crop_bbox(
    img: np.ndarray, cx: float, cy: float, bw: float, bh: float, padding: float
) -> np.ndarray | None:
    """Recorta una bbox normalizada de la imagen, con padding fraccional.
    Devuelve None si el recorte resultante es demasiado pequeño."""
    h, w = img.shape[:2]
    px = cx * w
    py = cy * h
    pw = bw * w * (1 + 2 * padding)
    ph = bh * h * (1 + 2 * padding)
    x1 = int(max(0, px - pw / 2))
    y1 = int(max(0, py - ph / 2))
    x2 = int(min(w, px + pw / 2))
    y2 = int(min(h, py + ph / 2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return img[y1:y2, x1:x2]


def _device(device: str) -> str | int:
    return 0 if device == "gpu" else "cpu"


def _torch_device(device: str) -> str:
    if device == "gpu" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
