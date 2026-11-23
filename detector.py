"""Local YOLO video analysis with transparent, heuristic incident scoring."""
from __future__ import annotations

import os
from collections import deque
from pathlib import Path

import cv2

DEFAULT_YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
BLIP_MODEL = os.getenv("BLIP_MODEL", "Salesforce/blip-image-captioning-base")
WEAPON_LABELS = {"knife", "scissors", "baseball bat"}
SUSPICIOUS_LABELS = WEAPON_LABELS | {"backpack", "handbag", "suitcase", "cell phone", "laptop"}


def _mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _load_captioner():
    """Load a small open model only when explicitly enabled; first run downloads weights."""
    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(BLIP_MODEL)
    model = BlipForConditionalGeneration.from_pretrained(BLIP_MODEL)
    model.eval()
    return processor, model, torch


def analyze_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    snapshot_path: str | Path | None = None,
    progress_callback=None,
    use_caption_model: bool = False,
) -> dict:
    """Analyze all readable frames and return metrics, summaries, and evidence paths.

    Threat percentage is a heuristic fraction of flagged frames, not a calibrated
    probability. YOLO's COCO weights do not reliably recognize weapons or crimes.
    """
    from ultralytics import YOLO

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    model = YOLO(DEFAULT_YOLO_MODEL)  # Ultralytics fetches its open weights on first use.
    captioner = _load_captioner() if use_caption_model else None
    writer = None
    frame_count = flagged_count = motion_flagged_count = 0
    peak_confidence = 0.0
    incident_start = incident_end = None
    incident_runs = []
    active_start = None
    best_snapshot = None
    keyframes = []
    recent_people = deque(maxlen=2)
    object_counts = {}
    width = height = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp = frame_count / fps
            height, width = frame.shape[:2]
            prediction = model.predict(frame, verbose=False, conf=0.30, imgsz=640)[0]
            boxes = prediction.boxes
            names = prediction.names
            detections = []
            person_centers = []
            max_object_conf = 0.0
            suspicious = []
            for box in boxes:
                cls_id = int(box.cls[0].item())
                label = str(names[cls_id]).lower()
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                detections.append((x1, y1, x2, y2, label, confidence))
                if label == "person":
                    person_centers.append(((x1 + x2) / 2 / width, (y1 + y2) / 2 / height))
                if label in SUSPICIOUS_LABELS:
                    suspicious.append((label, confidence, (x1, y1, x2, y2)))
                    object_counts[label] = object_counts.get(label, 0) + 1
                    max_object_conf = max(max_object_conf, confidence)

            rapid_motion = False
            if recent_people and person_centers:
                old_centers = recent_people[-1]
                nearest_motion = max((min(((x-a)**2 + (y-b)**2)**0.5 for a, b in old_centers) for x, y in person_centers), default=0)
                rapid_motion = nearest_motion > 0.12
            recent_people.append(person_centers)
            close_interaction = False
            if len(person_centers) >= 2:
                close_interaction = min(
                    ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
                    for i, a in enumerate(person_centers) for b in person_centers[i+1:]
                ) < 0.12

            flagged = bool(suspicious) or (rapid_motion and close_interaction)
            if rapid_motion and close_interaction:
                motion_flagged_count += 1
            confidence = max(max_object_conf, 0.45 + min(0.45, (0.2 if rapid_motion else 0) + (0.2 if close_interaction else 0))) if flagged else 0.0
            if flagged:
                flagged_count += 1
                peak_confidence = max(peak_confidence, confidence)
                if active_start is None:
                    active_start = timestamp
                    if incident_start is None:
                        incident_start = timestamp
                incident_end = timestamp
                if len(keyframes) < 12 or timestamp - keyframes[-1][0] >= 2.0:
                    keyframes.append((timestamp, frame.copy(), [x[0] for x in suspicious]))
                if best_snapshot is None or confidence > best_snapshot[0]:
                    best_snapshot = (confidence, frame.copy(), timestamp)
            elif active_start is not None:
                incident_runs.append((active_start, timestamp))
                active_start = None

            if frame_count % max(1, int(fps)) == 0:
                for x1, y1, x2, y2, label, conf in detections:
                    color = (0, 0, 255) if label in SUSPICIOUS_LABELS else (40, 200, 40)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(18, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)
            if output_path:
                if writer is None:
                    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                writer.write(frame)
            frame_count += 1
            if progress_callback and (frame_count % max(1, int(fps)) == 0):
                progress_callback(min(frame_count / total_hint, 1.0) if total_hint else 0.0)
        if active_start is not None:
            incident_runs.append((active_start, max(incident_end or active_start, (frame_count-1)/fps)))
    finally:
        cap.release()
        if writer:
            writer.release()

    if frame_count == 0:
        raise ValueError("The video contained no readable frames.")
    threat_percent = flagged_count / frame_count * 100.0
    if best_snapshot and snapshot_path:
        Path(snapshot_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot_path), best_snapshot[1])

    caption_text = ""
    if captioner and keyframes:
        processor, caption_model, torch = captioner
        captions = []
        for _, image, _ in keyframes[:4]:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            inputs = processor(images=rgb, return_tensors="pt")
            with torch.inference_mode():
                generated = caption_model.generate(**inputs, max_new_tokens=40)
            captions.append(processor.decode(generated[0], skip_special_tokens=True))
        caption_text = " ".join(captions)

    if flagged_count:
        labels = sorted(object_counts)
        motion_flagged = motion_flagged_count > 0
        descriptions = []
        if motion_flagged:
            descriptions.append("possible fight or physical altercation based on rapid movement and close proximity between people")
        if labels:
            descriptions.append("detected objects: " + ", ".join(labels))
        if caption_text:
            descriptions.append("image caption: " + caption_text)
        if not descriptions:
            descriptions.append("activity was flagged by the local screening heuristic")
        event_text = "; ".join(descriptions)
        summary = (f"Possible noteworthy activity detected. {event_text}. The scan flagged "
                   f"{flagged_count} of {frame_count} frames ({threat_percent:.1f}%). "
                   "This is an automated screening cue for human review, not a confirmed finding.")
    else:
        summary = (f"No noteworthy activity was flagged in {frame_count} analyzed frames. "
                   "The detector can miss actions and objects, so this does not rule out an incident.")
    return {
        "frame_count": frame_count, "duration_seconds": (frame_count-1) / fps,
        "threat_percent": threat_percent, "peak_confidence": peak_confidence,
        "incident_start": _mmss(incident_start) if incident_start is not None else None,
        "incident_end": _mmss(incident_end) if incident_end is not None else None,
        "incident_runs": [(_mmss(a), _mmss(b)) for a, b in incident_runs],
        "summary": summary,
        "possible_fight": bool(flagged_count and motion_flagged_count),
        "object_counts": object_counts,
        "keyframes": [(t, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), labs) for t, img, labs in keyframes],
        "video_path": str(output_path) if output_path and Path(output_path).exists() else None,
    }
