"""Local temporal activity screening: pose kinematics, VideoMAE and Qwen2-VL.

The Kinetics-400 VideoMAE checkpoint is a general action classifier, not a
violence-trained model. A result is therefore a review cue, never a finding.
"""
from __future__ import annotations

import os
import tempfile
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

POSE_MODEL_ID = os.getenv("YOLO_POSE_MODEL", "yolov8n-pose.pt")
WEAPON_MODEL_ID = os.getenv("WEAPON_MODEL", "")
ACTION_MODEL_ID = os.getenv("ACTION_MODEL", "MCG-NJU/videomae-base-finetuned-kinetics")
VLM_MODEL_ID = os.getenv("VLM_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
SAMPLE_FPS = float(os.getenv("ACTION_SAMPLE_FPS", "5"))
ACTION_THRESHOLD = float(os.getenv("ACTION_SCORE_THRESHOLD", "0.18"))
PROXIMITY_DISTANCE = float(os.getenv("PROXIMITY_DISTANCE", "0.18"))
LIMB_SPEED_THRESHOLD = float(os.getenv("LIMB_SPEED_THRESHOLD", "1.5"))
VIOLENT_TOKENS = (
    "fight", "fighting", "boxing", "wrestling", "kickboxing", "punch", "punching",
    "kick", "kicking", "slap", "slapping", "hitting", "martial arts", "karate",
    "judo", "headbutt", "brawl", "brawling", "choking", "strangling",
)
MIN_ACTION_RAM_GIB = float(os.getenv("MIN_ACTION_RAM_GIB", "4"))
MIN_VLM_RAM_GIB = float(os.getenv("MIN_VLM_RAM_GIB", "16"))
POSE_JOINTS = {"left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7,
               "right_elbow": 8, "left_wrist": 9, "right_wrist": 10,
               "left_hip": 11, "right_hip": 12, "left_knee": 13,
               "right_knee": 14, "left_ankle": 15, "right_ankle": 16}


@dataclass
class PersonPose:
    box: tuple[float, float, float, float]
    center: tuple[float, float]
    joints: dict[str, tuple[float, float]]


def _mmss(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 60:02d}:{total % 60:02d}"


def _available_ram_bytes() -> int:
    """Return the lower of host-available RAM and the current cgroup headroom."""
    import psutil

    available = int(psutil.virtual_memory().available)
    cgroup_files = (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
         Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    )
    for limit_path, usage_path in cgroup_files:
        try:
            raw_limit = limit_path.read_text(encoding="ascii").strip()
            if raw_limit == "max":
                continue
            limit, usage = int(raw_limit), int(usage_path.read_text(encoding="ascii").strip())
            # cgroup v1 uses a very large sentinel when there is no effective cap.
            if 0 < limit < 1 << 60:
                available = min(available, max(0, limit-usage))
                break
        except (OSError, ValueError):
            continue
    return available


def _iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-6)


def _pose_people(prediction, width: int, height: int, confidence: float = 0.25) -> list[PersonPose]:
    people = []
    if prediction.boxes is None or prediction.keypoints is None:
        return people
    boxes = prediction.boxes.xyxy.cpu().numpy()
    scores = prediction.boxes.conf.cpu().numpy()
    points = prediction.keypoints.xy.cpu().numpy()
    point_scores = getattr(prediction.keypoints, "conf", None)
    point_scores = point_scores.cpu().numpy() if point_scores is not None else None
    for index, box in enumerate(boxes):
        if float(scores[index]) < confidence:
            continue
        x1, y1, x2, y2 = map(float, box)
        joints = {}
        for name, joint_index in POSE_JOINTS.items():
            if joint_index >= len(points[index]):
                continue
            if point_scores is not None and float(point_scores[index][joint_index]) < 0.2:
                continue
            x, y = map(float, points[index][joint_index])
            if 0 <= x < width and 0 <= y < height:
                joints[name] = (x, y)
        people.append(PersonPose((x1, y1, x2, y2), ((x1+x2)/2, (y1+y2)/2), joints))
    return people


def _kinematic_score(people: list[PersonPose], previous: list[PersonPose],
                     width: int, height: int, dt: float) -> tuple[float, bool, list[tuple[tuple[int, int], tuple[int, int]]]]:
    """Measure close interaction and wrist/ankle speed, loosely tracking centers."""
    if len(people) < 2:
        return 0.0, False, []
    diagonal = max(float(np.hypot(width, height)), 1.0)
    proximity = 0.0
    interaction_pairs = []
    for i, person_a in enumerate(people):
        for person_b in people[i+1:]:
            distance = float(np.hypot(person_a.center[0]-person_b.center[0], person_a.center[1]-person_b.center[1])) / diagonal
            overlap = _iou(person_a.box, person_b.box)
            if overlap > 0.01 or distance < PROXIMITY_DISTANCE:
                interaction_pairs.append((person_a.center, person_b.center))
                proximity = max(proximity, max(min(overlap * 4.0, 1.0), 1.0-distance/PROXIMITY_DISTANCE))
    if not interaction_pairs:
        return 0.0, False, []

    max_speed = 0.0
    for person in people:
        if not previous:
            continue
        prior = min(previous, key=lambda old: np.hypot(person.center[0]-old.center[0], person.center[1]-old.center[1]))
        body_size = max(float(np.hypot(person.box[2]-person.box[0], person.box[3]-person.box[1])), 1.0)
        for joint in ("left_wrist", "right_wrist", "left_ankle", "right_ankle"):
            if joint in person.joints and joint in prior.joints:
                displacement = float(np.hypot(person.joints[joint][0]-prior.joints[joint][0], person.joints[joint][1]-prior.joints[joint][1]))
                max_speed = max(max_speed, displacement / body_size / max(dt, 1e-3))
    limb_motion = min(max_speed / (LIMB_SPEED_THRESHOLD * 2.0), 1.0)
    candidate = max_speed >= LIMB_SPEED_THRESHOLD
    score = proximity * limb_motion if candidate else 0.0
    lines = [(tuple(map(int, a)), tuple(map(int, b))) for a, b in interaction_pairs]
    return float(score), candidate, lines


def _load_action_model(device: str):
    import torch

    minimum_ram = MIN_ACTION_RAM_GIB * 1024**3
    if device == "cpu" and _available_ram_bytes() < minimum_ram:
        raise MemoryError(
            f"VideoMAE analysis needs at least {minimum_ram / 1024**3:.1f} GiB of free RAM. "
            "This hosted instance is too small; run the scan locally or use a larger instance."
        )
    # Import transformers only after the container-memory preflight. Importing
    # it eagerly can itself push a small Streamlit container into an OOM restart.
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    dtype = torch.float16 if device == "cuda" else torch.float32
    processor = VideoMAEImageProcessor.from_pretrained(ACTION_MODEL_ID)
    model = VideoMAEForVideoClassification.from_pretrained(
        ACTION_MODEL_ID, torch_dtype=dtype, low_cpu_mem_usage=True
    )
    model.eval().to(device)
    return processor, model, torch


def _classify_window(frames: list[np.ndarray], processor, model, torch, device: str) -> dict:
    rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames]
    inputs = processor(rgb, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        logits = model(**inputs).logits[0]
        probs = torch.softmax(logits, dim=-1)
    top_ids = torch.topk(probs, k=min(5, probs.numel())).indices.tolist()
    labels = model.config.id2label
    ranked = [{"label": str(labels[int(i)]), "score": float(probs[int(i)].item())} for i in top_ids]
    # Search every class for a target action, even if it is not in the top five.
    violent = [{"label": str(labels[i]), "score": float(probs[i].item())}
               for i in range(probs.numel())
               if any(token in str(labels[i]).lower() for token in VIOLENT_TOKENS)]
    best_violent = max(violent, key=lambda item: item["score"], default={"label": "", "score": 0.0})
    return {"top_action": ranked[0], "ranked_actions": ranked,
            "violence_action": best_violent["label"], "violence_score": best_violent["score"]}


def _load_vlm():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        required_memory = MIN_VLM_RAM_GIB * 1024**3
        available_memory = _available_ram_bytes()
        if available_memory < required_memory:
            raise MemoryError(
                "Qwen2-VL 2B CPU inference needs about 9 GiB of free RAM. "
                "Disable the optional storyboard summary or run locally on a larger machine."
            )
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    load_options = {"torch_dtype": torch.float16 if device == "cuda" else torch.float32}
    model = None
    if device == "cuda":
        try:
            model = Qwen2VLForConditionalGeneration.from_pretrained(
                VLM_MODEL_ID, device_map="auto", load_in_4bit=True
            )
        except Exception:
            model = None
    if model is None:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            VLM_MODEL_ID, low_cpu_mem_usage=True, **load_options
        ).to(device)
    return AutoProcessor.from_pretrained(VLM_MODEL_ID), model, torch, device


def _make_storyboard(frames: list[np.ndarray], path: str | Path | None = None) -> np.ndarray:
    if not frames:
        raise ValueError("No frames available for storyboard evidence.")
    indexes = np.linspace(0, len(frames)-1, 4).round().astype(int)
    tiles = []
    for index in indexes:
        frame = frames[int(index)]
        tile = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        cv2.putText(tile, f"Frame {int(index)+1}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(tile)
    storyboard = np.vstack((np.hstack(tiles[:2]), np.hstack(tiles[2:])))
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), storyboard):
            raise OSError(f"Could not write storyboard evidence to {path}.")
    return storyboard


def _summarize_storyboard(storyboard: np.ndarray) -> str:
    """Generate a local evidence summary; errors fall back to an explicit local summary."""
    try:
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        processor, model, torch, device = _load_vlm()
        image = Image.fromarray(cv2.cvtColor(storyboard, cv2.COLOR_BGR2RGB))
        prompt = (
            "Review this four-frame incident storyboard. Examine the sequence internally: "
            "describe visible body and limb posture, whether people appear to make physical contact, "
            "the change across frames, and uncertainty. Return only a concise factual screening "
            "summary with a cautious threat level. Do not infer identity, intent, or a crime. "
            "If the frames do not support an altercation, say so."
        )
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                           padding=True, return_tensors="pt").to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        answer = processor.batch_decode(
            output[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        if answer:
            return answer
    except Exception as exc:
        return f"Local storyboard summary unavailable ({type(exc).__name__}: {exc}). Review the attached four-frame evidence and model action labels."
    return "Local storyboard summary was empty. Review the attached four-frame evidence and model action labels."


def _incident_runs(mask: list[bool], timestamps: list[float], merge_gap_seconds: float = 0.0) -> list[dict]:
    runs = []
    start = None
    last = None
    max_gap = max(0.0, merge_gap_seconds)
    for index, flagged in enumerate(mask):
        if flagged:
            if start is None:
                start = index
            last = index
        elif start is not None and timestamps[index] - timestamps[last] > max_gap:
            runs.append({"start": _mmss(timestamps[start]), "end": _mmss(timestamps[last]),
                         "start_seconds": timestamps[start], "end_seconds": timestamps[last]})
            start = last = None
    if start is not None:
        runs.append({"start": _mmss(timestamps[start]), "end": _mmss(timestamps[last]),
                     "start_seconds": timestamps[start], "end_seconds": timestamps[last]})
    return runs


def analyze_video(video_path: str | Path | int, output_path: str | Path | None = None,
                  snapshot_path: str | Path | None = None, progress_callback=None,
                  use_caption_model: bool = False, *, live_seconds: int | None = None,
                  frame_callback: Callable | None = None, enable_vlm: bool = False,
                  action_threshold: float = ACTION_THRESHOLD) -> dict:
    """Run pose candidates, 16-frame VideoMAE windows and optional Qwen storyboard.

    Frame percentage counts every decoded frame whose pose candidate overlaps a
    target action label above ``action_threshold``. Webcam input may be an integer
    camera index; set ``live_seconds`` to bound the capture duration.
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        required_memory = MIN_ACTION_RAM_GIB * 1024**3
        available_memory = _available_ram_bytes()
        if available_memory < required_memory:
            raise RuntimeError(
                f"Video analysis needs at least {MIN_ACTION_RAM_GIB:.1f} GiB of free RAM; "
                f"this app container currently has {available_memory / 1024**3:.1f} GiB. "
                "The scan was stopped before loading AI models to keep the app online. "
                "Run it on a local computer or use a larger Streamlit instance."
            )

    # Delay the heavyweight Ultralytics import until after the cgroup RAM check.
    from ultralytics import YOLO

    live_input = isinstance(video_path, int)
    cap = cv2.VideoCapture(video_path if live_input else str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if not np.isfinite(fps) or fps <= 0:
        fps = 25.0
    total_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    try:
        pose_model = YOLO(POSE_MODEL_ID)
        weapon_model = YOLO(WEAPON_MODEL_ID) if WEAPON_MODEL_ID else None
        action_processor, action_model, torch = _load_action_model(device)
    except Exception as exc:
        cap.release()
        raise RuntimeError(f"Could not load local pose/action models: {exc}") from exc

    writer = raw_writer = None
    temp_video_dir = tempfile.TemporaryDirectory(prefix="temporal_scan_")
    raw_video_path = str(Path(temp_video_dir.name) / "decoded_source.avi")
    frame_count = 0
    sampled_frames: deque[np.ndarray] = deque(maxlen=16)
    sample_timestamps: deque[float] = deque(maxlen=16)
    candidate_flags: deque[bool] = deque(maxlen=16)
    sampled_count = 0
    next_window_start = 0
    windows = []
    best_window = None
    previous_people: list[PersonPose] = []
    frame_people: list[list[PersonPose]] = []
    frame_lines: list[list[tuple[tuple[int, int], tuple[int, int]]]] = []
    frame_candidates: list[bool] = []
    frame_timestamps: list[float] = []
    frame_weapon_scores: list[float] = []
    frame_weapon_labels: list[str] = []
    object_counts: dict[str, int] = {}
    best_weapon_evidence = None
    previous_sample_second = -1.0
    previous_frame_timestamp = -1.0
    live_action_label = ""
    live_action_score = 0.0
    started = time.monotonic()
    try:
        while True:
            if live_input and live_seconds and time.monotonic()-started >= live_seconds:
                break
            ok, frame = cap.read()
            if not ok:
                break
            reported_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
            reported_timestamp = reported_ms/1000.0
            timestamp = (reported_timestamp if np.isfinite(reported_timestamp) and
                         (frame_count == 0 or reported_timestamp > previous_frame_timestamp)
                         else frame_count/fps)
            previous_frame_timestamp = timestamp
            height, width = frame.shape[:2]
            if raw_writer is None:
                raw_writer = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
                if not raw_writer.isOpened():
                    raise OSError("Could not create temporary video buffer for analysis.")
            raw_writer.write(frame)
            prediction = pose_model.predict(frame, verbose=False, conf=0.25, imgsz=640)[0]
            people = _pose_people(prediction, width, height)
            dt = 1.0/fps
            score, candidate, lines = _kinematic_score(people, previous_people, width, height, dt)
            previous_people = people
            weapon_score = 0.0
            weapon_label = ""
            if weapon_model is not None:
                weapon_prediction = weapon_model.predict(frame, verbose=False, conf=0.35, imgsz=640)[0]
                names = weapon_prediction.names
                for detection in weapon_prediction.boxes:
                    label = str(names[int(detection.cls[0].item())]).lower()
                    confidence = float(detection.conf[0].item())
                    object_counts[label] = object_counts.get(label, 0)+1
                    if confidence > weapon_score:
                        weapon_score, weapon_label = confidence, label
                if weapon_score > (best_weapon_evidence[0] if best_weapon_evidence else 0.0):
                    best_weapon_evidence = (weapon_score, frame.copy())
            frame_people.append(people)
            frame_lines.append(lines)
            frame_candidates.append(candidate)
            frame_timestamps.append(timestamp)
            frame_weapon_scores.append(weapon_score)
            frame_weapon_labels.append(weapon_label)
            if timestamp - previous_sample_second >= 1.0/SAMPLE_FPS:
                sample = frame.copy()
                sampled_frames.append(sample)
                sample_timestamps.append(timestamp)
                candidate_flags.append(candidate)
                sampled_count += 1
                if len(sampled_frames) == 16 and sampled_count - 16 >= next_window_start:
                    window_start = sampled_count - 16
                    if any(candidate_flags):
                        classification = _classify_window(list(sampled_frames), action_processor, action_model, torch, device)
                        windows.append((sample_timestamps[0], sample_timestamps[-1], classification))
                        if classification["violence_score"] > live_action_score:
                            live_action_score = classification["violence_score"]
                            live_action_label = classification["violence_action"]
                        if best_window is None or classification["violence_score"] > best_window[0]:
                            best_window = (classification["violence_score"], list(sampled_frames))
                    next_window_start = window_start + 8
                previous_sample_second = timestamp
            frame_count += 1
            if frame_callback and frame_count % max(1, int(fps/5)) == 0:
                preview = frame.copy()
                for person in people:
                    x1, y1, x2, y2 = map(int, person.box)
                    cv2.rectangle(preview, (x1, y1), (x2, y2), (30, 220, 30), 2)
                    for joint, point in person.joints.items():
                        cv2.circle(preview, tuple(map(int, point)), 4, (0, 220, 255), -1)
                    for left, right in (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                                        ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                                        ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
                                        ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
                                        ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                                        ("right_hip", "right_knee"), ("right_knee", "right_ankle")):
                        if left in person.joints and right in person.joints:
                            cv2.line(preview, tuple(map(int, person.joints[left])), tuple(map(int, person.joints[right])), (255, 180, 0), 2)
                for point_a, point_b in lines:
                    cv2.line(preview, point_a, point_b, (0, 130, 255), 2)
                live_malicious = candidate and live_action_score >= action_threshold
                status = "POSSIBLE ALTERCATION" if live_malicious else ("CANDIDATE WINDOW" if candidate else "MONITORING")
                color = (0, 0, 255) if live_malicious else (0, 130, 255) if candidate else (25, 120, 25)
                cv2.rectangle(preview, (0, 0), (width, 38), color, -1)
                cv2.putText(preview, f"{status} | {live_action_label or 'pose cue'} {max(score, live_action_score):.0%} | {timestamp:.1f}s",
                            (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                frame_callback(preview, timestamp, max(score, live_action_score), status)
            if progress_callback and total_hint:
                progress_callback(min(frame_count/total_hint, 0.99), "Pose and kinematic screening")
            elif progress_callback and live_input and live_seconds:
                progress_callback(min(timestamp/live_seconds, 0.99), "Live pose and temporal analysis")
    finally:
        cap.release()
        if raw_writer:
            raw_writer.release()
    if not frame_count:
        raise ValueError("The video contained no readable frames.")
    if len(sampled_frames) < 16 and any(candidate_flags):
        # Very short clips use edge repetition to retain the classifier's fixed input size.
        while len(sampled_frames) < 16:
            sampled_frames.append(sampled_frames[-1].copy())
            sample_timestamps.append(sample_timestamps[-1])
            candidate_flags.append(False)
        classification = _classify_window(list(sampled_frames), action_processor, action_model, torch, device)
        windows.append((sample_timestamps[0], sample_timestamps[-1], classification))
        best_window = (classification["violence_score"], list(sampled_frames))
    if progress_callback:
        progress_callback(0.99, "Temporal action classification complete")

    frame_risk = list(frame_weapon_scores)
    frame_action = [f"weapon: {label}" if label else "" for label in frame_weapon_labels]
    action_labels = {}
    top_action_labels = {}
    for begin_t, end_t, classification in windows:
        top = classification["top_action"]
        top_action_labels[top["label"]] = max(top_action_labels.get(top["label"], 0.0), top["score"])
        label = classification["violence_action"]
        confidence = classification["violence_score"]
        if confidence < action_threshold:
            continue
        action_labels[label] = max(action_labels.get(label, 0.0), confidence)
        finish_t = end_t + 1.0/max(SAMPLE_FPS, 1.0)
        for index in range(frame_count):
            timestamp = frame_timestamps[index]
            if begin_t <= timestamp <= finish_t and frame_candidates[index]:
                frame_risk[index] = max(frame_risk[index], confidence)
                frame_action[index] = label

    malicious_mask = [risk >= action_threshold for risk in frame_risk]
    malicious_count = sum(malicious_mask)
    peak_confidence = max(frame_risk, default=0.0)
    threat_percent = 100.0*malicious_count/frame_count
    runs = _incident_runs(malicious_mask, frame_timestamps, merge_gap_seconds=0.0)
    labels = sorted(action_labels, key=lambda label: action_labels[label], reverse=True)
    possible_fight = any(any(token in label.lower() for token in VIOLENT_TOKENS) for label in labels)

    # Save an annotated copy with skeletons, proximity lines and a per-frame status banner.
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        source_cap = cv2.VideoCapture(raw_video_path)
        ok, first_frame = source_cap.read()
        if not ok:
            source_cap.release()
            raise OSError("Could not read the temporary video buffer for annotated rendering.")
        height, width = first_frame.shape[:2]
        writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            source_cap.release()
            raise OSError(f"Could not create annotated video: {output_path}")
        index = 0
        source_frame = first_frame
        while index < frame_count:
            frame = source_frame.copy()
            for person in frame_people[index]:
                x1, y1, x2, y2 = map(int, person.box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 220, 30), 2)
                for name, point in person.joints.items():
                    cv2.circle(frame, tuple(map(int, point)), 4, (0, 210, 255), -1)
                for left, right in (("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
                                    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
                                    ("left_shoulder", "right_shoulder"), ("left_shoulder", "left_hip"),
                                    ("right_shoulder", "right_hip"), ("left_hip", "right_hip"),
                                    ("left_hip", "left_knee"), ("left_knee", "left_ankle"),
                                    ("right_hip", "right_knee"), ("right_knee", "right_ankle")):
                    if left in person.joints and right in person.joints:
                        cv2.line(frame, tuple(map(int, person.joints[left])), tuple(map(int, person.joints[right])), (255, 180, 0), 2)
            for point_a, point_b in frame_lines[index]:
                cv2.line(frame, point_a, point_b, (0, 130, 255), 2)
            status = "POSSIBLE ALTERCATION" if malicious_mask[index] else ("CANDIDATE WINDOW" if frame_candidates[index] else "MONITORING")
            color = (0, 0, 255) if malicious_mask[index] else (0, 165, 255) if frame_candidates[index] else (25, 140, 25)
            cv2.rectangle(frame, (0, 0), (width, 42), color, -1)
            label = frame_action[index] or status
            cv2.putText(frame, f"{status} | {label} | {frame_risk[index]:.0%}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            writer.write(frame)
            index += 1
            if index < frame_count:
                ok, source_frame = source_cap.read()
                if not ok:
                    break
        writer.release()
        source_cap.release()
        writer = None

    storyboard_path = None
    summary = ("No target physical-action label was confirmed in pose-triggered temporal windows. "
               + ("Custom weapon detector labels: " + ", ".join(sorted(object_counts)) + ". " if object_counts else "")
               + ("Candidate-window action labels: " + ", ".join(
                   f"{label} ({score:.0%})" for label, score in sorted(top_action_labels.items(), key=lambda item:item[1], reverse=True)[:3]) + ". " if top_action_labels else "")
               + "This can miss events; inspect the video and model limitations.")
    if malicious_count:
        storyboard_frames = best_window[1] if action_labels and best_window else ([best_weapon_evidence[1]] if best_weapon_evidence else list(sampled_frames))
        storyboard_path = str(snapshot_path) if snapshot_path else None
        storyboard = _make_storyboard(storyboard_frames, storyboard_path)
        vlm_summary = _summarize_storyboard(storyboard) if enable_vlm and use_caption_model else ""
        actions = ", ".join(f"{name} ({score:.0%})" for name, score in sorted(action_labels.items(), key=lambda item:item[1], reverse=True))
        if action_labels:
            description = (f"Possible physical altercation cue. VideoMAE action label(s): {actions}. "
                           f"{malicious_count}/{frame_count} frames ({threat_percent:.1f}%) overlap a close-contact pose candidate and target action window.")
        else:
            description = (f"Custom weapon detector flagged: {', '.join(sorted(object_counts)) or 'an object'}. "
                           f"{malicious_count}/{frame_count} frames ({threat_percent:.1f}%) were flagged by the object model.")
        summary = (description + f" Peak threat score: {peak_confidence:.1%}. "
                   + (f"Storyboard review: {vlm_summary}" if vlm_summary else "Review the attached four-frame storyboard; local VLM summarization is disabled or unavailable."))

    if output_path:
        output_path = str(output_path) if Path(output_path).is_file() else None
    if progress_callback:
        progress_callback(1.0, "Analysis complete")
    temp_video_dir.cleanup()
    return {
        "frame_count": frame_count, "duration_seconds": frame_timestamps[-1],
        "threat_percent": threat_percent, "peak_confidence": peak_confidence,
        "incident_start": runs[0]["start"] if runs else None,
        "incident_end": runs[-1]["end"] if runs else None,
        "incident_runs": runs, "summary": summary, "object_counts": object_counts,
        "action_labels": action_labels, "top_action_labels": top_action_labels,
        "possible_fight": possible_fight,
        "malicious_frames": malicious_count, "storyboard_path": storyboard_path,
        "keyframes": [], "video_path": output_path,
        "device": device, "action_model": ACTION_MODEL_ID,
    }
