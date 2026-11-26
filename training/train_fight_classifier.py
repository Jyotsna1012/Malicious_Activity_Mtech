"""Fine-tune, evaluate, export and INT8-quantize the compact fight model.

Manifest columns: video_path,label,group_id,start_seconds,end_seconds
Labels are exactly fight or non_fight. group_id must identify the source video
or recording session so clips from one source cannot leak across data splits.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fight_model import build_model

LABELS = ["non_fight", "fight"]
FRAMES = 8
IMAGE_SIZE = 160
MOTION_GATE_DEFAULT = 0.20
MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"video_path", "label", "group_id"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs these columns: {', '.join(sorted(required))}")
        rows = []
        for row_number, row in enumerate(reader, start=2):
            label = row["label"].strip().lower()
            if label not in LABELS:
                raise ValueError(f"Row {row_number}: label must be one of {LABELS}, got {label!r}")
            video = Path(row["video_path"].strip())
            if not video.is_absolute():
                video = (path.parent / video).resolve()
            if not video.is_file():
                raise FileNotFoundError(f"Row {row_number}: video not found: {video}")
            rows.append({
                "video_path": video,
                "label": label,
                "group_id": row["group_id"].strip(),
                "start_seconds": float(row.get("start_seconds") or 0),
                "end_seconds": float(row.get("end_seconds") or 0),
            })
    if not rows or {row["label"] for row in rows} != set(LABELS):
        raise ValueError("Manifest must contain clips from both fight and non_fight classes.")
    if any(not row["group_id"] for row in rows):
        raise ValueError("Every clip needs a non-empty group_id representing its source video/session.")
    return rows


class VideoClips(Dataset):
    def __init__(self, rows: list[dict], augment: bool = False,
                 motion_gate: float = MOTION_GATE_DEFAULT):
        self.rows = rows
        self.augment = augment
        self.motion_gate = motion_gate

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        capture = cv2.VideoCapture(str(row["video_path"]))
        if not capture.isOpened():
            raise OSError(f"Could not open training clip: {row['video_path']}")
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
        total_frames = max(1, int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 1))
        duration = total_frames / max(fps, 1e-3)
        start = max(0.0, min(row["start_seconds"], max(0.0, duration - 0.1)))
        end = row["end_seconds"] if row["end_seconds"] > start else duration
        end = max(start + 0.1, min(end, duration))
        clip_duration = end - start
        offset = random.random() if self.augment else 0.5
        fractions = (np.arange(FRAMES, dtype=np.float32) + offset) / FRAMES
        timestamps = start + fractions * clip_duration
        images = []
        last_frame = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        for timestamp in timestamps:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp * 1000))
            ok, frame = capture.read()
            if ok:
                last_frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
            images.append(last_frame.copy())
        capture.release()

        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in images]
        flow_scores = []
        for previous, current in zip(gray_frames, gray_frames[1:]):
            flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 2, 15, 2, 5, 1.1, 0)
            magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
            flow_scores.append(float(np.percentile(magnitude, 90)))
        candidate = bool(flow_scores and max(flow_scores) >= self.motion_gate)

        flip = self.augment and random.random() < 0.5
        brightness = random.uniform(0.85, 1.15) if self.augment else 1.0
        clip = []
        for frame in images:
            if flip:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = np.clip(rgb * brightness, 0.0, 1.0)
            clip.append(((rgb - MEAN) / STD).transpose(2, 0, 1))
        tensor = torch.from_numpy(np.stack(clip).astype(np.float32))
        target = LABELS.index(row["label"])
        return tensor, torch.tensor(target, dtype=torch.long), torch.tensor(candidate, dtype=torch.bool)


def split_by_group(rows: list[dict], seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = np.arange(len(rows))
    groups = np.asarray([row["group_id"] for row in rows])
    labels = np.asarray([row["label"] for row in rows])
    if len(np.unique(groups)) < 5:
        raise ValueError("Need at least 5 distinct source groups for group-safe train/validation/test splits.")
    for attempt in range(100):
        random_state = seed + attempt
        first = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_val, test = next(first.split(indices, labels, groups))
        second = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state + 1000)
        train_rel, val_rel = next(second.split(train_val, labels[train_val], groups[train_val]))
        train, val = train_val[train_rel], train_val[val_rel]
        if all(set(labels[part]) == set(LABELS) for part in (train, val, test)):
            return train.tolist(), val.tolist(), test.tolist()
    raise ValueError(
        "Could not create group-safe splits containing both classes. Add more independent source videos "
        "for each class; do not split clips from one source across train and test."
    )


def logits_for(model, loader, device):
    model.eval()
    all_logits, all_targets, all_candidates = [], [], []
    with torch.inference_mode():
        for clips, targets, candidates in loader:
            all_logits.append(model(clips.to(device)).cpu().numpy())
            all_targets.extend(targets.numpy().tolist())
            all_candidates.extend(candidates.numpy().tolist())
    return (np.concatenate(all_logits), np.asarray(all_targets, dtype=np.int64),
            np.asarray(all_candidates, dtype=bool))


def metrics(probabilities: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
    tp = int(np.sum((predictions == 1) & (targets == 1)))
    tn = int(np.sum((predictions == 0) & (targets == 0)))
    fp = int(np.sum((predictions == 1) & (targets == 0)))
    fn = int(np.sum((predictions == 0) & (targets == 1)))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"threshold": float(threshold), "accuracy": (tp + tn) / max(len(targets), 1),
            "fight_precision": precision, "fight_recall": recall, "fight_f1": f1,
            "false_positive_rate": fp / max(fp + tn, 1),
            "confusion_matrix_tn_fp_fn_tp": [tn, fp, fn, tp]}


def apply_motion_gate(probabilities: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Mirror deployment: windows rejected by optical flow cannot be positive."""
    effective = probabilities.copy()
    effective[~candidates, 1] = 0.0
    effective[~candidates, 0] = 1.0
    return effective


def choose_threshold(probabilities: np.ndarray, targets: np.ndarray, target_recall: float) -> float:
    candidates = np.arange(0.2, 0.901, 0.025)
    scored = [(metrics(probabilities, targets, float(value)), float(value)) for value in candidates]
    feasible = [(m["fight_precision"], m["fight_f1"], threshold)
                for m, threshold in scored if m["fight_recall"] >= target_recall]
    if feasible:
        return max(feasible)[2]
    # If validation cannot meet the requested recall, choose the highest-recall
    # threshold and report the shortfall rather than hiding missed fights.
    return max((m["fight_recall"], m["fight_f1"], threshold)
               for m, threshold in scored)[2]


def export_onnx(model, destination: Path) -> None:
    model.eval().cpu()
    sample = torch.zeros(1, FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32)
    torch.onnx.export(model, sample, str(destination), input_names=["clips"],
                      output_names=["logits"], dynamic_axes={"clips": {0: "batch"}, "logits": {0: "batch"}},
                      opset_version=17, do_constant_folding=True, dynamo=False)


def onnx_probabilities(model_path: Path, dataset: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    probs, targets, candidates = [], [], []
    for clips, target, candidate in DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0):
        logits = session.run(None, {name: clips.numpy()})[0][0].astype(np.float64)
        logits -= logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        probs.append(p)
        targets.append(int(target.item()))
        candidates.append(bool(candidate.item()))
    return np.asarray(probs), np.asarray(targets), np.asarray(candidates, dtype=bool)


class CalibrationReader:
    def __init__(self, dataset: Dataset, max_clips: int = 128):
        self.dataset = dataset
        self.max_clips = min(len(dataset), max_clips)
        self.index = 0

    def get_next(self):
        if self.index >= self.max_clips:
            return None
        clip, *_ = self.dataset[self.index]
        self.index += 1
        return {"clips": clip.unsqueeze(0).numpy()}

    def rewind(self):
        self.index = 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-recall", type=float, default=0.90,
                        help="Choose the highest-precision validation threshold meeting this fight recall when possible.")
    parser.add_argument("--motion-gate", type=float, default=MOTION_GATE_DEFAULT,
                        help="Optical-flow candidate threshold; validate end-to-end recall when tuning.")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet MobileNetV3 weights.")
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    rows = load_manifest(args.manifest.resolve())
    train_ids, val_ids, test_ids = split_by_group(rows, args.seed)
    train_ds = VideoClips([rows[i] for i in train_ids], augment=True, motion_gate=args.motion_gate)
    calibration_ds = VideoClips([rows[i] for i in train_ids], augment=False, motion_gate=args.motion_gate)
    val_ds = VideoClips([rows[i] for i in val_ids], motion_gate=args.motion_gate)
    test_ds = VideoClips([rows[i] for i in test_ids], motion_gate=args.motion_gate)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(pretrained=not args.no_pretrained).to(device)
    labels = np.asarray([rows[i]["label"] for i in train_ids])
    counts = np.asarray([np.sum(labels == label) for label in LABELS], dtype=np.float32)
    class_weights = torch.tensor(len(labels) / (2.0 * np.maximum(counts, 1)), device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    best_f1 = -1.0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "fight_classifier.best.pth"

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for clips, targets, _candidates in train_loader:
            clips, targets = clips.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(clips), targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        val_logits, val_targets, val_candidates = logits_for(model, val_loader, device)
        val_probs = apply_motion_gate(torch.softmax(torch.from_numpy(val_logits), dim=1).numpy(), val_candidates)
        threshold = choose_threshold(val_probs, val_targets, args.target_recall)
        val_metrics = metrics(val_probs, val_targets, threshold)
        print(f"epoch {epoch:02d}/{args.epochs} loss={np.mean(losses):.4f} "
              f"val_recall={val_metrics['fight_recall']:.3f} val_f1={val_metrics['fight_f1']:.3f} "
              f"threshold={threshold:.3f}", flush=True)
        if val_metrics["fight_f1"] > best_f1:
            best_f1 = val_metrics["fight_f1"]
            torch.save(model.state_dict(), checkpoint_path)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.cpu().eval()
    float_path = args.output_dir / "fight_detector.fp32.onnx"
    int8_path = args.output_dir / "fight_detector.int8.onnx"
    export_onnx(model, float_path)

    val_fp32, val_y, val_candidates = onnx_probabilities(float_path, val_ds)
    val_fp32 = apply_motion_gate(val_fp32, val_candidates)
    threshold_fp32 = choose_threshold(val_fp32, val_y, args.target_recall)
    test_fp32, test_y, test_candidates = onnx_probabilities(float_path, test_ds)
    test_fp32 = apply_motion_gate(test_fp32, test_candidates)
    report = {"fp32_validation": metrics(val_fp32, val_y, threshold_fp32),
              "fp32_test": metrics(test_fp32, test_y, threshold_fp32),
              "motion_gate_validation_recall": float(np.sum(val_candidates & (val_y == 1)) / max(np.sum(val_y == 1), 1)),
              "motion_gate_test_recall": float(np.sum(test_candidates & (test_y == 1)) / max(np.sum(test_y == 1), 1))}

    try:
        from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

        quantize_static(str(float_path), str(int8_path), CalibrationReader(calibration_ds),
                        quant_format=QuantFormat.QDQ, activation_type=QuantType.QUInt8,
                        weight_type=QuantType.QInt8, per_channel=True,
                        op_types_to_quantize=["Conv", "MatMul", "Gemm"])
        val_int8, val_y_int8, val_candidates_int8 = onnx_probabilities(int8_path, val_ds)
        val_int8 = apply_motion_gate(val_int8, val_candidates_int8)
        threshold_int8 = choose_threshold(val_int8, val_y_int8, args.target_recall)
        test_int8, test_y_int8, test_candidates_int8 = onnx_probabilities(int8_path, test_ds)
        test_int8 = apply_motion_gate(test_int8, test_candidates_int8)
        report.update({"int8_validation": metrics(val_int8, val_y_int8, threshold_int8),
                       "int8_test": metrics(test_int8, test_y_int8, threshold_int8),
                       "int8_bytes": int8_path.stat().st_size,
                       "fp32_bytes": float_path.stat().st_size})
        int8_recall = report["int8_validation"]["fight_recall"]
        float_recall = report["fp32_validation"]["fight_recall"]
        int8_f1 = report["int8_validation"]["fight_f1"]
        float_f1 = report["fp32_validation"]["fight_f1"]
        use_int8 = int8_recall >= float_recall - 0.02 and int8_f1 >= float_f1 - 0.03
        deploy_path = int8_path if use_int8 else float_path
        if not use_int8:
            print("INT8 validation dropped beyond tolerance; selecting the FP32 ONNX artifact.", flush=True)
    except Exception as exc:
        print(f"INT8 export failed ({type(exc).__name__}: {exc}); using FP32 ONNX.", flush=True)
        deploy_path = float_path
        threshold_int8 = threshold_fp32

    deployment_threshold = threshold_int8 if deploy_path == int8_path else threshold_fp32
    runtime_path = args.output_dir / "fight_detector.onnx"
    runtime_metadata = args.output_dir / "fight_detector.metadata.json"
    shutil.copy2(deploy_path, runtime_path)
    metadata = {"labels": LABELS, "frames": FRAMES, "input_size": IMAGE_SIZE,
                "thresholds": {"fp32": threshold_fp32, "int8": threshold_int8,
                               "deployment": deployment_threshold},
                "sample_fps": 3, "motion_gate": args.motion_gate,
                "validation_and_test": report}
    runtime_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nModel artifact for deployment: {runtime_path} (selected from {deploy_path.name})")
    print(f"Metadata and held-out metrics: {runtime_metadata}")
    print(json.dumps(report, indent=2))
    print("Review fight recall and false-positive rate. Do not deploy the INT8 artifact if its held-out metrics are unacceptable.")


if __name__ == "__main__":
    main()
