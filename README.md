# AI-Powered Malicious Activity Detection & Alert System

Local video screening using pose kinematics, a temporal action classifier and a storyboard vision-language model. Inference does not call paid APIs or send video to a hosted model service. The project includes a Streamlit customer portal, SQLite accounts and quotas, incident history, webcam analysis and optional SMTP alerts.

## 1. Project structure and dependencies

```text
.
├── app.py              # Streamlit auth, analyzer, plans, and incident history
├── database.py         # SQLite users, subscriptions, quotas, and incidents
├── detector.py         # YOLO pose, kinematics, VideoMAE windows, and Qwen VLM
├── alerts.py           # HTML SMTP alerts and local console fallback
├── main.py             # Earlier standalone research prototype (not used by app.py)
├── Emotion.py          # Earlier emotion-analysis experiment (not used by app.py)
├── requirements.txt
├── .env.example
└── README.md
```

`requirements.txt` includes Ultralytics, Transformers, PyTorch/torchvision, OpenCV, Streamlit, SQLAlchemy, Pillow, NumPy and `qwen-vl-utils`. PyTorch wheels differ by CPU/CUDA and must be installed from the official selector before the remaining requirements.

### Model and license note

All inference is local and uses downloadable model weights. The default [VideoMAE Kinetics-400 checkpoint](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics) is tagged **CC BY-NC 4.0** (non-commercial). It is not an OSI open-source license for commercial use. The [Qwen2-VL model card](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) lists Apache 2.0. Review the model and dataset licenses before redistribution or commercial deployment. YOLOv8 software/weights have their own Ultralytics license terms.

The default VideoMAE model is trained on Kinetics-400, not a dedicated fight-recognition dataset. Its action vocabulary includes only the classes in that dataset; this implementation can score matching labels such as boxing or wrestling when available, but it cannot reliably distinguish assault from sport or guarantee detection of punching, slapping, or other acts. For dissertation claims about fight detection, fine-tune and evaluate a classifier on a licensed, representative fight/non-fight dataset and report precision, recall, calibration, and subgroup performance. Treat the current outputs as cues for human review.

## 2. Database and subscription manager (`database.py`)

SQLite stores `Users` (email and salted scrypt password hashes), `Subscriptions` (plan, activity and alert threshold), and `IncidentLogs` (summary, action labels, exact incident runs, storyboard path, alert status and metrics). `init_db()` creates the schema and adds new incident fields to existing SQLite installations.

| Plan | Monthly scans | Live webcam | Alert threshold |
|---|---:|---|---|
| Basic | 5 | No | 20% default |
| Pro | 50 | Yes | 20% default |
| Enterprise | Unlimited | Yes | Configurable |

Tier selection is a simulated checkout and does not process payments. The monthly quota counts completed scan records. Set the Enterprise threshold in the Subscription page.

## 3. Temporal detection engine (`detector.py`)

1. **Spatial and pose cues:** `yolov8n-pose.pt` estimates people and COCO keypoints. The engine estimates inter-person box IoU/proximity, tracks wrists/ankles between adjacent frames, and selects candidate windows when proximity and rapid limb motion coincide. Set `WEAPON_MODEL` to an optional custom weapon detector checkpoint; the default pose model does not detect weapons.
2. **Temporal action classification:** frames are sampled at 5 FPS by default and passed as 16-frame, stride-8 windows to local VideoMAE only when a pose candidate is present. A frame contributes to the malicious activity percentage when its pose candidate overlaps a target temporal action score above `ACTION_SCORE_THRESHOLD`. The percentage is `malicious_frames / decoded_frames * 100`; score values are model softmax outputs and are not calibrated probabilities. CPU inference checks host and container (cgroup) memory and requires 4 GiB free before loading the model. Smaller hosted instances receive an actionable error instead of exhausting the app process.
3. **Storyboard summary:** the highest-scoring action window is reduced to a four-frame 2×2 storyboard and can be passed to Qwen2-VL-2B-Instruct locally. This option is off by default because the 2B model needs substantial RAM; CPU loading is skipped when less than 9 GiB is available. CUDA uses 4-bit loading when supported and otherwise FP16. If Qwen cannot load, the detector preserves the evidence and returns a fallback explanation.

The result includes frame count, percentage, peak action score, target and top candidate labels, per-incident start/end timestamps (`MM:SS` and fractional seconds), an annotated MP4, and a four-frame JPEG storyboard. Video frames are spooled to a temporary local file during inference to keep decoded video out of long-lived memory. The temporary file is removed after the scan.

Live webcam access reads camera 0 on the machine hosting Streamlit. Pro/Enterprise users see a live pose/proximity preview; temporal classification updates the threat banner after a complete 16-frame window is available. A remote browser's camera is not accessed.

## 4. Alert dispatcher (`alerts.py`)

An email is sent when `threat_percent` is strictly greater than the user threshold (20% by default). The HTML email includes action labels, metrics and incident windows, and attaches the local storyboard image. Configure `SMTP_HOST`, `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD` and `ALERT_FROM` in the process environment. For Gmail, use an App Password. If credentials are missing or delivery fails, the alert is logged locally without exposing credentials. Alerts below threshold are not sent.

## 5. Streamlit dashboard (`app.py`)

The portal provides registration/login, plan selection, MP4/AVI upload, plan-gated webcam analysis, local live preview, progress, annotated playback, threat metrics/timeline, configurable Enterprise thresholds, and incident history with storyboard evidence. The server webcam is only available when Streamlit runs on a host with an attached camera.

Keep `.env` secrets and `user_data/` private. This is a dissertation prototype, not a hardened public SaaS deployment. Add production-grade session management, CSRF protections, rate limiting, private object storage, retention controls, HTTPS, and operational monitoring before exposing it to untrusted users.

## 6. Setup and execution

Use Python 3.10 or 3.11. Create and activate a virtual environment, then install a PyTorch build matching the machine:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install --upgrade pip
# Select CPU or CUDA wheel at https://pytorch.org/get-started/locally/
python -m pip install torch torchvision
python -m pip install -r requirements.txt
```

Copy `.env.example` to `.env`. `app.py` loads that file with `python-dotenv`; deployment environments can set the same values directly. Keep credentials out of source control.

Pre-cache model weights before disconnecting from the network:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
python -c "from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification; m='MCG-NJU/videomae-base-finetuned-kinetics'; VideoMAEImageProcessor.from_pretrained(m); VideoMAEForVideoClassification.from_pretrained(m)"
python -c "from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; m='Qwen/Qwen2-VL-2B-Instruct'; AutoProcessor.from_pretrained(m); Qwen2VLForConditionalGeneration.from_pretrained(m)"
```

Start the application:

```bash
streamlit run app.py
```

First downloads require internet access; all subsequent inference is local. Set `HF_HUB_OFFLINE=1` after the models are cached to make Hugging Face loading fail fast instead of attempting a network request. Use `HF_HOME` to select the local model cache directory. Set `ACTION_MODEL`, `VLM_MODEL`, `YOLO_POSE_MODEL` or `WEAPON_MODEL` in the environment to select local model IDs/paths.
