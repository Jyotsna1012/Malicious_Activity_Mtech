"""Low-memory video screening with a trained, quantized temporal classifier.

The deployment process uses OpenCV and ONNX Runtime only. PyTorch is confined
to the separate model-training environment.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("FIGHT_MODEL_PATH", "models/fight_detector.onnx"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
WINDOW_FRAMES = int(os.getenv("FIGHT_WINDOW_FRAMES", "8"))
SAMPLE_FPS = float(os.getenv("FIGHT_SAMPLE_FPS", "3"))
WINDOW_STRIDE = int(os.getenv("FIGHT_WINDOW_STRIDE", "4"))
INPUT_SIZE = int(os.getenv("FIGHT_INPUT_SIZE", "160"))
MOTION_GATE = float(os.getenv("FIGHT_MOTION_GATE", "0.20"))


class ModelUnavailableError(RuntimeError):
    """Raised when no locally trained fight classifier is installed."""


def _load_classifier():
    if not MODEL_PATH.is_file():
        raise ModelUnavailableError(
            f"The trained fight model is not installed ({MODEL_PATH}). "
            "Train it with training/train_fight_classifier.py, then place the "
            "quantized ONNX file at this path. See README.md for the data format and commands."
        )
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ModelUnavailableError(
            "ONNX Runtime is missing. Install the deployment packages from requirements.txt."
        ) from exc

    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(os.getenv("ONNX_THREADS", "2")))
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(MODEL_PATH), sess_options=options, providers=["CPUExecutionProvider"]
    )
    metadata_path = MODEL_PATH.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    labels = metadata.get("labels", ["non_fight", "fight"])
    thresholds = metadata.get("thresholds", {})
    threshold = float(os.getenv("FIGHT_THRESHOLD", thresholds.get("deployment", thresholds.get("int8", 0.5))))
    motion_gate = float(os.getenv("FIGHT_MOTION_GATE", metadata.get("motion_gate", MOTION_GATE)))
    return session, session.get_inputs()[0].name, labels, threshold, motion_gate


def _prepare_window(frames: list[np.ndarray]) -> np.ndarray:
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 1, 3)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 1, 3)
    prepared = []
    for frame in frames:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        prepared.append(((rgb.astype(np.float32) / 255.0 - mean[0, 0]) / std[0, 0]).transpose(2, 0, 1))
    return np.expand_dims(np.stack(prepared), axis=0).astype(np.float32)


def _predict(session, input_name: str, labels: list[str], frames: list[np.ndarray]) -> tuple[str, float]:
    logits = session.run(None, {input_name: _prepare_window(frames)})[0][0].astype(np.float64)
    logits -= logits.max()
    probabilities = np.exp(logits) / np.exp(logits).sum()
    index = int(np.argmax(probabilities))
    return labels[index], float(probabilities[index])


def _flow_score(previous_gray: np.ndarray | None, frame: np.ndarray) -> tuple[np.ndarray, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
    if previous_gray is None:
        return gray, 0.0
    flow = cv2.calcOpticalFlowFarneback(previous_gray, gray, None, 0.5, 2, 15, 2, 5, 1.1, 0)
    magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    # A high percentile preserves localized fast limb motion even when most of
    # the scene is static. Tune the gate against validation footage for recall.
    return gray, float(np.percentile(magnitude, 90))


def _incident_runs(events: list[dict], merge_gap: float = 0.75) -> list[dict]:
    if not events:
        return []
    events = sorted(events, key=lambda item: item["start_seconds"])
    merged = [dict(events[0])]
    for event in events[1:]:
        current = merged[-1]
        if event["start_seconds"] <= current["end_seconds"] + merge_gap:
            current["end_seconds"] = max(current["end_seconds"], event["end_seconds"])
            current["confidence"] = max(current["confidence"], event["confidence"])
        else:
            merged.append(dict(event))
    for event in merged:
        event["start"] = _mmss(event["start_seconds"])
        event["end"] = _mmss(event["end_seconds"])
    return merged


def _mmss(seconds: float) -> str:
    value = max(0, int(seconds))
    return f"{value // 60:02d}:{value % 60:02d}"


def _write_storyboard(frames: list[np.ndarray], path: str | Path | None) -> np.ndarray:
    if not frames:
        raise ValueError("No frames were available to create incident evidence.")
    indexes = np.linspace(0, len(frames) - 1, 4).round().astype(int)
    tiles = []
    for index, frame_index in enumerate(indexes, start=1):
        tile = cv2.resize(frames[int(frame_index)], (480, 270), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"Frame {index}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)
        tiles.append(tile)
    storyboard = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), storyboard):
            raise OSError(f"Could not write storyboard to {path}")
    return storyboard


def _annotate_video(source: str | Path | int, output_path: str | Path | None,
                    events: list[dict], frame_callback=None, live_seconds: int | None = None,
                    on_progress=None) -> tuple[int, float]:
    capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = None
    frame_count = 0
    started = time.monotonic()
    try:
        while True:
            if isinstance(source, int) and live_seconds and time.monotonic() - started >= live_seconds:
                break
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_count / fps
            is_fight = any(event["start_seconds"] <= timestamp <= event["end_seconds"] for event in events)
            if output_path:
                if writer is None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    output_width = min(frame.shape[1], 960)
                    output_height = round(frame.shape[0] * output_width / frame.shape[1])
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, (output_width, output_height))
                    if not writer.isOpened():
                        raise OSError(f"Could not create annotated video at {output_path}")
                frame = cv2.resize(frame, (output_width, output_height), interpolation=cv2.INTER_AREA)
                color = (0, 0, 255) if is_fight else (30, 160, 50)
                banner = "POSSIBLE FIGHT - REVIEW" if is_fight else "No fight window flagged"
                cv2.rectangle(frame, (0, 0), (frame.shape[1], 38), color, -1)
                cv2.putText(frame, f"{banner}  {timestamp:06.2f}s", (10, 26),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                writer.write(frame)
            if frame_callback:
                score = max((e["confidence"] for e in events
                             if e["start_seconds"] <= timestamp <= e["end_seconds"]), default=0.0)
                frame_callback(frame, timestamp, score,
                               "Possible fight" if is_fight else "No fight window flagged")
            frame_count += 1
            if on_progress and frame_count % 30 == 0:
                value = frame_count / total_hint if total_hint else min(frame_count / (fps * 60), 0.95)
                on_progress(min(value, 0.99), f"Writing annotated video: {frame_count} frames")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    return frame_count, fps


def analyze_video(video_path: str | Path | int, output_path: str | Path | None = None,
                  snapshot_path: str | Path | None = None, progress_callback=None,
                  use_caption_model: bool = False, *, live_seconds: int | None = None,
                  frame_callback=None, enable_vlm: bool = False,
                  action_threshold: float | None = None) -> dict:
    """Analyze video in a bounded-memory sliding window using a local ONNX model."""
    del use_caption_model, enable_vlm  # Kept for compatibility with older callers.
    session, input_name, labels, default_threshold, motion_gate = _load_classifier()
    threshold = default_threshold if action_threshold is None else float(action_threshold)
    capture = cv2.VideoCapture(video_path if isinstance(video_path, int) else str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_hint = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sample_interval = max(1, round(fps / max(SAMPLE_FPS, 0.1)))
    live_temp = tempfile.TemporaryDirectory(prefix="fight_live_") if isinstance(video_path, int) and output_path else None
    live_raw_path = Path(live_temp.name) / "capture.avi" if live_temp else None
    live_writer = None
    sampled = deque(maxlen=WINDOW_FRAMES)
    motion_scores = deque(maxlen=WINDOW_FRAMES)
    previous_gray = None
    frame_index = 0
    sampled_count = 0
    last_inference_at = -WINDOW_STRIDE
    best_window: list[np.ndarray] = []
    best_confidence = 0.0
    events: list[dict] = []
    window_count = 0
    candidate_window_count = 0
    started = time.monotonic()
    try:
        while True:
            if isinstance(video_path, int) and live_seconds and time.monotonic() - started >= live_seconds:
                break
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = frame_index / fps
            if live_raw_path and live_writer is None:
                live_writer = cv2.VideoWriter(str(live_raw_path), cv2.VideoWriter_fourcc(*"MJPG"),
                                              fps, (frame.shape[1], frame.shape[0]))
                if not live_writer.isOpened():
                    raise OSError("Could not buffer the live webcam feed for annotated playback.")
            if live_writer is not None:
                live_writer.write(frame)
            if isinstance(video_path, int) and frame_callback:
                frame_callback(frame.copy(), timestamp, best_confidence,
                               "Possible fight candidate" if events else "Screening live frames")
            if frame_index % sample_interval == 0:
                previous_gray, motion = _flow_score(previous_gray, frame)
                # Keep only small copies of the active window. Full-resolution
                # source frames are never accumulated in process memory.
                sampled.append(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA))
                motion_scores.append(motion)
                sampled_count += 1
                if len(sampled) == WINDOW_FRAMES and sampled_count - last_inference_at >= WINDOW_STRIDE:
                    last_inference_at = sampled_count
                    window_count += 1
                    flow_candidate = max(motion_scores) >= motion_gate
                    if flow_candidate:
                        candidate_window_count += 1
                        window = list(sampled)
                        label, confidence = _predict(session, input_name, labels, window)
                        if label == "fight" and confidence >= threshold:
                            start_seconds = max(0.0, timestamp - (WINDOW_FRAMES - 1) / SAMPLE_FPS)
                            event = {"start_seconds": start_seconds, "end_seconds": timestamp,
                                     "confidence": confidence, "label": label}
                            events.append(event)
                            if confidence > best_confidence:
                                best_confidence, best_window = confidence, list(window)
                if progress_callback and total_hint and frame_index % max(sample_interval * 8, 1) == 0:
                    progress_callback(min(frame_index / total_hint, 0.95),
                                      f"Screening video: {timestamp:.1f}s / {total_hint / fps:.1f}s")
            frame_index += 1
    finally:
        capture.release()
        if live_writer is not None:
            live_writer.release()

    if frame_index == 0:
        raise ValueError(f"No video frames could be read from: {video_path}")
    incidents = _incident_runs(events)
    total_duration = frame_index / fps
    flagged_duration = sum(max(0.0, item["end_seconds"] - item["start_seconds"]) for item in incidents)
    threat_percent = min(100.0, flagged_duration / max(total_duration, 1e-6) * 100.0)
    malicious_frames = min(frame_index, round(frame_index * threat_percent / 100.0))
    snapshot = _write_storyboard(best_window or [], snapshot_path) if best_window else None

    if output_path:
        annotation_source = live_raw_path if live_raw_path else video_path
        _annotate_video(annotation_source, output_path, incidents, on_progress=progress_callback)
    if live_temp:
        live_temp.cleanup()

    if incidents:
        summary = (f"Possible fight detected from {incidents[0]['start']} to "
                   f"{incidents[-1]['end']} (peak model score {best_confidence:.0%}). "
                   "Review the storyboard and original video; this is an automated screening cue.")
        action_labels = {"fight": best_confidence}
        no_candidate_windows = False
    elif candidate_window_count == 0:
        summary = ("The motion screen found no candidate windows to classify. This does not rule out a fight; "
                   "review the video or lower FIGHT_MOTION_GATE and evaluate it on validation clips.")
        action_labels = {}
        no_candidate_windows = True
    else:
        summary = "No fight was flagged by the trained local model. Review the video if context suggests otherwise."
        action_labels = {}
        no_candidate_windows = False
    if progress_callback:
        progress_callback(1.0, "Analysis complete")
    return {
        "possible_fight": bool(incidents), "summary": summary,
        "no_candidate_windows": no_candidate_windows,
        "window_count": window_count, "candidate_window_count": candidate_window_count,
        "frame_count": frame_index, "malicious_frames": malicious_frames,
        "threat_percent": threat_percent, "peak_confidence": best_confidence,
        "incident_start": incidents[0]["start"] if incidents else None,
        "incident_end": incidents[-1]["end"] if incidents else None,
        "incident_runs": incidents, "action_labels": action_labels,
        "top_action_labels": {"fight": best_confidence} if incidents else {},
        "object_counts": {}, "action_model": f"Compact MobileNetV3-GRU ONNX ({MODEL_PATH.name})",
        "device": "CPU / ONNX Runtime", "video_path": str(output_path) if output_path else "",
        "storyboard_path": str(snapshot_path) if snapshot is not None and snapshot_path else None,
    }
