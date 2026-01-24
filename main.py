import cv2
from collections import Counter
import numpy as np
import mediapipe as mp
from fer import FER
import warnings
warnings.filterwarnings("ignore")

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

emotion_detector = FER(mtcnn=True)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=300, varThreshold=80, detectShadows=False
)

# ==================================================
# VIDEO INPUT (20 seconds limit)
# ==================================================
video_path = "sampleVideos/1/R_11.mp4"
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 25

MAX_SECONDS = 20
MAX_FRAMES = int(fps * MAX_SECONDS)
frame_counter = 0

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
    if action != "Fight / Attack":
        return "NO_FIGHT"

    # SELF PRACTICE only if exactly ONE person
    if person_count == 1:
        return "SELF_PRACTICE"

    # Multiple persons → NEVER self practice
    if emotion in ["fear", "sad"]:
        return "PUNISHMENT"
    if emotion == "angry":
        return "ATTACK"

    return "ATTACK"

# --------------------------------------------------
# RISK + THREAT SCORE
# --------------------------------------------------
def assess_risk_and_score(fight_type, action, emotion, person_count):
    score = 0

    if action == "Fight / Attack":
        score += 40
    if emotion in ["angry", "fear", "sad"]:
        score += 30
    if person_count >= 2:
        score += 20
    if action == "Running" and emotion in ["angry", "fear", "sad"]:
        score += 10
    if action in ["Fall Detected", "Sleeping / Fallen"]:
        score += 25

    score = min(score, 100)

    if score >= 70:
        return "MALICIOUS", score
    elif score >= 25:
        return "SUSPICIOUS", score
    else:
        return "SAFE", score

# ==================================================
# MAIN LOOP
# ==================================================
while cap.isOpened() and frame_counter < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        ACTION_LABEL, prev_center, prev_height = detect_action(
            results.pose_landmarks.landmark, prev_center, prev_height, w, h
        )

    detections = emotion_detector.detect_emotions(frame)
    dominant_emotion = "neutral"
    face_count = len(detections)

    for face in detections:
        (x, y, fw, fh) = face["box"]
        emotions = face["emotions"]
        face_emotion = max(emotions, key=emotions.get)
        dominant_emotion = face_emotion

        cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 1)
        cv2.putText(frame, face_emotion, (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

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

    # DISPLAY (UNCHANGED)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 90), (0, 0, 0), -1)
    cv2.putText(frame, f"Action: {ACTION_LABEL}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,0), 1)
    cv2.putText(frame, f"Emotion: {dominant_emotion}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,255), 1)
    cv2.putText(frame, f"Fight Type: {fight_type}", (10, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,200,0), 1)
    cv2.putText(frame, f"Risk: {risk}", (10, 76),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0,0,255) if risk=="MALICIOUS"
                else (0,255,255) if risk=="SUSPICIOUS"
                else (0,255,0), 1)

    cv2.imshow("Surveillance – Fight Intent Analysis", frame)
    frame_counter += 1

    if cv2.waitKey(25) & 0xFF == ord("q"):
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
cv2.destroyAllWindows()

# ==================================================
# FINAL CONCLUSION (FORCED CONSISTENCY)
# ==================================================
max_threat = max(threat_score_log)
idx = threat_score_log.index(max_threat)

final_fight_type = fight_type_log[idx]

# 🚫 NEVER allow SELF_PRACTICE if multiple persons seen
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
print("-----------------------------------------------------")
print("LOGIC APPLIED             : Multi-person evaluation")
print("REASON                    : Action + Emotion based")
print("=====================================================\n")

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
