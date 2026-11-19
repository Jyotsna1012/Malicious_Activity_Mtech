import base64
import argparse
import json
import os
import subprocess
from collections import Counter

import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from openai import OpenAI
import warnings
warnings.filterwarnings("ignore")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.environ.get(
    "APP_CONFIG_FILE", os.path.join(BASE_DIR, "config.json")
)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

app_config = {}
if os.path.isfile(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as config_file:
            app_config = json.load(config_file)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file: {CONFIG_PATH}") from exc

if not OPENAI_API_KEY:
    OPENAI_API_KEY = str(app_config.get("OPENAI_API_KEY", "")).strip()
if not OPENAI_API_KEY:
    raise ValueError(
        "Set OPENAI_API_KEY via the OPENAI_API_KEY environment variable "
        f"or in the config file: {CONFIG_PATH}"
    )

# ==================================================
# INITIALIZE MODELS
# ==================================================
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

EMOTION_LABELS = ("anger", "disgust", "sad", "happiness", "surprise")
emotion_model_path = os.path.join(BASE_DIR, "FacialExpression_weights.keras")
if not os.path.isfile(emotion_model_path):
    raise FileNotFoundError(f"Trained emotion model not found: {emotion_model_path}")
emotion_model = tf.keras.models.load_model(emotion_model_path, compile=False)

face_detector = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.5,
)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=80, detectShadows=False
)

# ==================================================
# VIDEO INPUT (20 seconds limit)
# ==================================================
default_video_path = os.path.join(BASE_DIR, "sampleVideos", "1", "video2.mp4")
# Also support the sample video being placed under the project's venv folder.
if not os.path.isfile(default_video_path):
    venv_sample_path = os.path.join(BASE_DIR, "venv", "sampleVideos", "1", "video2.mp4")
    if os.path.isfile(venv_sample_path):
        default_video_path = venv_sample_path
parser = argparse.ArgumentParser(description="Analyze a video with CV and OpenAI vision.")
parser.add_argument(
    "video",
    nargs="?",
    default=default_video_path,
    help="Video file to analyze (defaults to sampleVideos/1/video2.mp4).",
)
parser.add_argument(
    "--no-preview",
    action="store_true",
    help="Process without opening the desktop preview window.",
)
args = parser.parse_args()
show_preview = not args.no_preview
video_path = os.path.abspath(args.video)
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise FileNotFoundError(
        f"Could not open input video: {video_path}\n"
        "Pass a readable video path, for example: python main.py path\\to\\video.mp4"
    )

output_video_path = os.path.splitext(video_path)[0] + "_annotated.mp4"

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

MAX_SECONDS = 20
MAX_FRAMES = min(int(fps * MAX_SECONDS), 540)
frame_counter = 0
SAMPLE_INTERVAL_SECONDS = 1.0
next_sample_time = 0.0
sampled_frames = []
sampled_frame_signals = []

# ==================================================
# GLOBAL VARIABLES
# ==================================================
prev_center = None
prev_height = None
ACTION_LABEL = "Normal"

risk_log = []
threat_score_log = []
emotion_log = []
action_log = []
fight_type_log = []
person_count_log = []
face_count_log = []

# ==================================================
# HELPER FUNCTIONS
# ==================================================
def lm_xy(landmarks, idx, w, h):
    lm = landmarks[idx]
    return np.array([int(lm.x * w), int(lm.y * h)])

# --------------------------------------------------
# ACTION DETECTION
# --------------------------------------------------
def detect_action(landmarks, prev_center, prev_height, w, h):
    head = lm_xy(landmarks, 0, w, h)
    l_hand = lm_xy(landmarks, 15, w, h)
    r_hand = lm_xy(landmarks, 16, w, h)
    l_foot = lm_xy(landmarks, 27, w, h)
    r_foot = lm_xy(landmarks, 28, w, h)

    center = (head + l_foot + r_foot) / 3
    height = abs(head[1] - max(l_foot[1], r_foot[1]))
    width = abs(l_foot[0] - r_foot[0])

    if prev_height is not None and (prev_height - height) > 90:
        return "Fall Detected", center, height

    speed = 0
    if prev_center is not None:
        speed = np.linalg.norm(center - prev_center)
        if speed > 45:
            return "Running", center, height

    hand_span = np.linalg.norm(l_hand - r_hand)
    foot_span = np.linalg.norm(l_foot - r_foot)

    if speed > 25 and (hand_span > 120 or foot_span > 140):
        return "Fight / Attack", center, height

    if height < width * 0.6:
        return "Sleeping / Fallen", center, height

    return "Normal", center, height

# --------------------------------------------------
# PEOPLE ESTIMATION
# --------------------------------------------------
def estimate_people_count(frame):
    fg = bg_subtractor.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    count = 0
    for c in contours:
        if 2000 < cv2.contourArea(c) < 20000:
            count += 1

    return min(count, 8)

# --------------------------------------------------
# FIGHT TYPE (STRICT)
# --------------------------------------------------
def classify_fight_type(action, emotion, person_count):
    # print(f"{action}  {emotion}  {person_count}")
    if action != "Fight / Attack":
        if action == "Sleeping / Fallen":
            return "Fallen"
        elif person_count <= 1:
            return "SELF_PRACTICE"  # SELF PRACTICE only if exactly ONE person   
        else:
            return "NO_FIGHT"
    if action == "Fight / Attack" :
        if person_count <= 1:
            return "SELF_PRACTICE"
        

    # Multiple persons → NEVER self practice
    if emotion in ["fear", "sad"]:
        return "PUNISHMENT"
    if emotion == "anger":
        return "ATTACK"

    return "ATTACK"

# --------------------------------------------------
# RISK + THREAT SCORE
# --------------------------------------------------
def assess_risk_and_score(fight_type, action, emotion, person_count):
    score = 0

    if action == "Fight / Attack":
        score += 25
    if emotion in ["anger", "fear", "sad"]:
        score += 10
    if person_count >= 2:
        score += 20
    if action == "Running" and emotion in ["anger", "fear", "sad"]:
        score += 10
    if action in ["Sleeping / Fallen"]:
        score += 40
    if action in ["Fall Detected"]:
        score += 10    

    score = min(score, 100)

    if score >= 70:
        return "MALICIOUS", score
    elif score > 25:
        return "SUSPICIOUS", score
    else:
        return "SAFE", score


def analyze_video_with_llm(frames, signals):
    """Ask a vision model for a cautious scene assessment from sampled frames."""
    if not frames:
        raise ValueError("No video frames were available for LLM analysis.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    content = [{
        "type": "input_text",
        "text": (
            "Review these chronologically ordered video frames and the accompanying "
            "computer-vision summaries. Classify only observable events, not a person's "
            "intent, character, or guilt. Do not infer aggression from facial emotion alone. "
            "For every supplied frame, return exactly one concise but specific observation "
            "of at most 25 words, in chronological order. Describe visible people, posture, "
            "movement, relative positions, and clearly visible physical interaction; say "
            "when a detail is unclear instead of guessing. Treat local computer-vision cues "
            "as fallible hints and describe only what the image supports. Then provide a "
            "clear overall summary in 2 to 4 sentences. Do not omit supplied frames. "
            "Use POSSIBLE_PHYSICAL_AGGRESSION only when visible physical interaction supports "
            "it; use FALL_OR_DISTRESS for a visible fall or apparent distress; use "
            "NO_CLEAR_PHYSICAL_AGGRESSION when evidence does not support aggression; use "
            "UNCLEAR when the frames are insufficient. Return concise evidence grounded in "
            "the frames, mention uncertainty, and output JSON matching the required schema.\n\n"
            f"Video duration analyzed: {signals['duration_seconds']:.1f} seconds.\n"
            f"Sampled frames: {len(frames)}.\n"
            f"Aggregated action labels: {json.dumps(signals['actions'])}.\n"
            f"Aggregated emotion labels: {json.dumps(signals['emotions'])}.\n"
            f"Observed face counts: {json.dumps(signals['face_counts'])}.\n"
            "Per-frame computer-vision cues follow with each corresponding image."
        ),
    }]
    for frame_index, (timestamp, image_data) in enumerate(frames):
        frame_cue_text = ""
        if frame_index < len(signals.get("frame_cues", [])):
            cue = signals["frame_cues"][frame_index]
            frame_cue_text = (
                f" Local CV cues: action={cue['action']}; emotion={cue['emotion']}; "
                f"estimated people={cue['people_count']}; risk={cue['risk']}."
            )
        content.append({
            "type": "input_text",
            "text": f"Frame at {timestamp:.1f} seconds.{frame_cue_text}",
        })
        content.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{image_data}",
            "detail": "high",
        })

    schema = {
        "type": "object",
        "properties": {
            "assessment": {
                "type": "string",
                "enum": [
                    "POSSIBLE_PHYSICAL_AGGRESSION",
                    "NO_CLEAR_PHYSICAL_AGGRESSION",
                    "FALL_OR_DISTRESS",
                    "UNCLEAR",
                ],
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "summary": {"type": "string"},
            "frame_summaries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "timestamp_seconds": {"type": "number", "minimum": 0},
                        "observation": {"type": "string"},
                    },
                    "required": ["timestamp_seconds", "observation"],
                    "additionalProperties": False,
                },
            },
            "evidence": {"type": "array", "items": {"type": "string"}},
            "uncertainties": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "assessment", "confidence", "summary", "frame_summaries",
            "evidence", "uncertainties",
        ],
        "additionalProperties": False,
    }
    response = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-6-luna"),
        input=[{"role": "user", "content": content}],
        text={
            "format": {
                "type": "json_schema",
                "name": "video_scene_assessment",
                "strict": True,
                "schema": schema,
            }
        },
        reasoning={"effort": "low"},
        max_output_tokens=4000,
        store=False,
    )
    assessment = json.loads(response.output_text)
    usage = getattr(response, "usage", None)
    assessment["api_usage"] = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    return assessment

# ==================================================
# MAIN LOOP
# ==================================================
preview_window = "Surveillance - Fight Intent Analysis"
if show_preview:
    cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
preview_max_width = 960
preview_max_height = 650

while cap.isOpened() and frame_counter < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    video_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    if video_time <= 0:
        video_time = frame_counter / fps
    sampled_image_data = None
    if video_time >= next_sample_time:
        sample = frame
        sample_height, sample_width = sample.shape[:2]
        if sample_width > 640:
            resized_height = round(sample_height * 640 / sample_width)
            sample = cv2.resize(sample, (640, resized_height))
        encoded_ok, encoded_frame = cv2.imencode(
            ".jpg", sample, [cv2.IMWRITE_JPEG_QUALITY, 75]
        )
        if encoded_ok:
            sampled_image_data = base64.b64encode(encoded_frame).decode("ascii")
        next_sample_time = video_time + SAMPLE_INTERVAL_SECONDS

    h, w, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        ACTION_LABEL, prev_center, prev_height = detect_action(
            results.pose_landmarks.landmark, prev_center, prev_height, w, h
        )

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detector.process(rgb_frame)
    face_rois = []
    face_boxes = []
    if face_results.detections:
        for detection in face_results.detections:
            relative_box = detection.location_data.relative_bounding_box
            x1 = max(0, int(relative_box.xmin * w))
            y1 = max(0, int(relative_box.ymin * h))
            x2 = min(w, int((relative_box.xmin + relative_box.width) * w))
            y2 = min(h, int((relative_box.ymin + relative_box.height) * h))
            if x2 <= x1 or y2 <= y1:
                continue

            face_crop = frame[y1:y2, x1:x2]
            gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            resized_face = cv2.resize(
                gray_face, (96, 96), interpolation=cv2.INTER_CUBIC
            )
            face_rois.append(resized_face.astype(np.float32) / 255.0)
            face_boxes.append((x1, y1, x2 - x1, y2 - y1))

    dominant_emotion = "neutral"
    face_count = len(face_rois)
    if face_rois:
        face_inputs = np.stack(face_rois, axis=0)[..., np.newaxis]
        emotion_scores = emotion_model(face_inputs, training=False).numpy()
        strongest_face_confidence = -1.0
        for (x, y, fw, fh), scores in zip(face_boxes, emotion_scores):
            emotion_index = int(np.argmax(scores))
            face_emotion = EMOTION_LABELS[emotion_index]
            confidence = float(scores[emotion_index])
            if confidence > strongest_face_confidence:
                dominant_emotion = face_emotion
                strongest_face_confidence = confidence
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{face_emotion} {confidence:.0%}",
                (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    motion_count = estimate_people_count(frame)

    if face_count > 0:
        person_count = face_count
    else:
        # Prevent 1 person being counted as many
        person_count = min(motion_count, 1)

    fight_type = classify_fight_type(ACTION_LABEL, dominant_emotion, person_count)
    risk, threat_score = assess_risk_and_score(
        fight_type, ACTION_LABEL, dominant_emotion, person_count
    )

    risk_log.append(risk)
    threat_score_log.append(threat_score)
    emotion_log.append(dominant_emotion)
    action_log.append(ACTION_LABEL)
    fight_type_log.append(fight_type)
    person_count_log.append(person_count)
    face_count_log.append(face_count)

    if sampled_image_data is not None:
        sampled_frames.append((video_time, sampled_image_data))
        sampled_frame_signals.append({
            "action": ACTION_LABEL,
            "emotion": dominant_emotion,
            "people_count": person_count,
            "risk": risk,
        })

    # Larger labels remain readable in both the output video and smaller preview.
    overlay_font = max(0.65, min(1.0, frame.shape[1] / 900.0))
    overlay_thickness = 2
    overlay_line_height = int(38 * overlay_font)
    overlay_margin = int(12 * overlay_font)
    overlay_height = overlay_line_height * 4 + overlay_margin * 2
    cv2.rectangle(frame, (0, 0), (frame.shape[1], overlay_height), (0, 0, 0), -1)
    cv2.putText(frame, f"Action: {ACTION_LABEL}", (overlay_margin, overlay_margin + overlay_line_height),
                cv2.FONT_HERSHEY_SIMPLEX, overlay_font, (255,255,0), overlay_thickness)
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (overlay_margin, overlay_margin + overlay_line_height * 2),
                cv2.FONT_HERSHEY_SIMPLEX, overlay_font, (0,255,255), overlay_thickness)
    cv2.putText(frame, f"Fight Type: {fight_type}", (overlay_margin, overlay_margin + overlay_line_height * 3),
                cv2.FONT_HERSHEY_SIMPLEX, overlay_font, (255,200,0), overlay_thickness)
    cv2.putText(frame, f"Risk: {risk}", (overlay_margin, overlay_margin + overlay_line_height * 4),
                cv2.FONT_HERSHEY_SIMPLEX, overlay_font,
                (0,0,255) if risk=="MALICIOUS"
                else (0,255,255) if risk=="SUSPICIOUS"
                else (0,255,0), overlay_thickness)

    if frame_counter == 0:
        height, width = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            output_video_path, fourcc, fps, (width, height)
        )
        if not video_writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {output_video_path}")
    video_writer.write(frame)

    if show_preview:
        # Scale only the on-screen preview; the saved video keeps the source size.
        preview_scale = min(
            1.0,
            preview_max_width / frame.shape[1],
            preview_max_height / frame.shape[0],
        )
        preview_frame = cv2.resize(
            frame,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_AREA,
        )
        cv2.imshow(preview_window, preview_frame)
    frame_counter += 1
    if frame_counter % max(1, int(fps)) == 0:
        print(f"Processed {frame_counter} frames...", flush=True)

    if show_preview and cv2.waitKey(25) & 0xFF == ord("q"):
        break
    
# print("\n================ FRAME BY FRAME LOGS ================")

# print("\nACTION LOG:")
# for i, a in enumerate(action_log):
#     print(f"Frame {i+1}: {a}")

# print("\nEMOTION LOG:")
# for i, e in enumerate(emotion_log):
#     print(f"Frame {i+1}: {e}")

# print("\nTHREAT SCORE LOG:")
# for i, t in enumerate(threat_score_log):
#     print(f"Frame {i+1}: {t}")

# print("=====================================================\n")
       
cap.release()
if "video_writer" in globals():
    video_writer.release()
if show_preview:
    cv2.destroyAllWindows()

# Encode a browser-compatible H.264 MP4 before the app exposes the video.
h264_path = os.path.splitext(output_video_path)[0] + "_h264.mp4"
try:
    import imageio_ffmpeg

    subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-i", output_video_path,
            "-c:v", "libx264",
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            "-an",
            "-f", "mp4",
            "-loglevel", "error",
            h264_path,
        ],
        check=True,
        capture_output=True,
    )
    if not os.path.isfile(h264_path) or os.path.getsize(h264_path) == 0:
        raise RuntimeError("FFmpeg produced an empty H.264 video.")
    os.replace(h264_path, output_video_path)
    print("Video transcoded to H.264 for browser playback.", flush=True)
except Exception as exc:
    if os.path.exists(h264_path):
        try:
            os.remove(h264_path)
        except OSError:
            pass
    raise RuntimeError(
        f"Could not encode the annotated video for browser playback: {exc}"
    ) from exc

print(f"Annotated video saved to: {output_video_path}", flush=True)

if not action_log:
    raise RuntimeError("The video contained no readable frames to analyze.")

try:
    llm_assessment = analyze_video_with_llm(
        sampled_frames,
        {
            "duration_seconds": frame_counter / fps,
            "actions": dict(Counter(action_log)),
            "emotions": dict(Counter(emotion_log)),
            "face_counts": dict(Counter(face_count_log)),
            "frame_cues": sampled_frame_signals,
        },
    )
except Exception as exc:
    print(f"LLM assessment skipped: {type(exc).__name__}: {exc}", flush=True)
    error_text = str(exc).lower()
    if "credit_balance_exhausted" in error_text or "insufficient_quota" in error_text:
        vision_error = (
            "OpenAI vision analysis could not run because the API account for this key "
            "has no credits. Add API credits, then reprocess the video."
        )
    else:
        vision_error = (
            f"OpenAI vision analysis could not run ({type(exc).__name__}). "
            "Frame descriptions and the visual summary are unavailable."
        )
    llm_assessment = {
        "assessment": "VISION_UNAVAILABLE",
        "confidence": None,
        "summary": "",
        "frame_summaries": [],
        "evidence": [],
        "uncertainties": [vision_error],
    }
else:
    vision_error = None

# ==================================================
# FINAL CONCLUSION (FORCED CONSISTENCY)
# ==================================================
max_threat = max(threat_score_log)
frame_activity_score = round(float(np.mean(threat_score_log)))
vision_aggression_score = 0
if llm_assessment["assessment"] == "POSSIBLE_PHYSICAL_AGGRESSION":
    vision_aggression_score = round(llm_assessment["confidence"] * 100)
malicious_activity_percent = max(frame_activity_score, vision_aggression_score)
idx = threat_score_log.index(max_threat)

final_fight_type = fight_type_log[idx]

# 🚫 NEVER allow SELF_ PRACTICE if multiple persons seen
if final_fight_type == "SELF_PRACTICE" and max(person_count_log) >= 2:
    if emotion_log[idx] in ["fear", "sad"]:
        final_fight_type = "PUNISHMENT"
    else:
        final_fight_type = "ATTACK"

print("\n================ FINAL VIDEO ANALYSIS ================")
print(f"VIDEO CLASSIFICATION RESULT : {risk_log[idx]}")
print("-----------------------------------------------------")
print(f"Max Faces Detected         : {max(face_count_log)}")
print(f"Estimated People Count    : {int(np.median(person_count_log))}")
print(f"Dominant Action           : {action_log[idx]}")
print(f"Dominant Emotion          : {emotion_log[idx]}")
print(f"Final Fight Type          : {final_fight_type}")
print(f"Threat Score (Max)        : {max_threat} / 100")
print(f"Frame Activity Score: {frame_activity_score}%")
print(f"Vision Assessment: {llm_assessment['assessment']}")
if llm_assessment["confidence"] is None:
    print("Vision Confidence: unavailable")
else:
    print(f"Vision Confidence: {llm_assessment['confidence']:.0%}")
print(f"Malicious Activity Score: {malicious_activity_percent}%")
if vision_error:
    print("VISION_ERROR_JSON: " + json.dumps(vision_error, ensure_ascii=False))
print("FRAME_SUMMARIES_JSON: " + json.dumps(
    llm_assessment.get("frame_summaries", []), ensure_ascii=False
))
print("VIDEO_SUMMARY_JSON: " + json.dumps(
    llm_assessment["summary"], ensure_ascii=False
))
if llm_assessment.get("api_usage"):
    print("VISION_USAGE_JSON: " + json.dumps(llm_assessment["api_usage"]))
print("-----------------------------------------------------")
print("LOGIC APPLIED             : Multi-person evaluation")
print("REASON                    : Action + Emotion based")
print("=====================================================\n")

print("\n================ OPENAI VISION ASSESSMENT ================")
print(f"Assessment                : {llm_assessment['assessment']}")
if llm_assessment["confidence"] is None:
    print("Confidence                : unavailable")
else:
    print(f"Confidence                : {llm_assessment['confidence']:.0%}")
print(f"Summary                   : {llm_assessment['summary'] or 'Not generated because vision analysis was unavailable.'}")
print("Evidence:")
for item in llm_assessment["evidence"]:
    print(f"  - {item}")
print("Uncertainties:")
for item in llm_assessment["uncertainties"]:
    print(f"  - {item}")
print("===========================================================\n")

print("\n================ PERCENTAGE SUMMARY (TABLE) ================")

total_frames = len(action_log)

# ---------- ACTION TABLE ----------
print("\nACTION SUMMARY")
print(f"{'Action':20s} | {'Count':6s} | {'Percent':8s}")
print("-" * 40)
action_counts = Counter(action_log)
for action, count in action_counts.items():
    percent = (count / total_frames) * 100
    print(f"{action:20s} | {count:6d} | {percent:7.2f}%")

# ---------- EMOTION TABLE ----------
print("\nEMOTION SUMMARY")
print(f"{'Emotion':20s} | {'Count':6s} | {'Percent':8s}")
print("-" * 40)
emotion_counts = Counter(emotion_log)
for emotion, count in emotion_counts.items():
    percent = (count / total_frames) * 100
    print(f"{emotion:20s} | {count:6d} | {percent:7.2f}%")

# ---------- THREAT TABLE ----------
print("\nTHREAT LEVEL SUMMARY")
print(f"{'Threat':20s} | {'Count':6s} | {'Percent':8s}")
print("-" * 40)
risk_counts = Counter(risk_log)
for risk, count in risk_counts.items():
    percent = (count / total_frames) * 100
    print(f"{risk:20s} | {count:6d} | {percent:7.2f}%")

print("============================================================\n")
