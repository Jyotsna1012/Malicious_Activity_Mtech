# AI-Powered Malicious Activity Detection & Alert System

A dissertation demonstration for local video screening, approximate incident metrics, optional local captions, SQLite accounts and scan quotas, and optional email alerts. The dashboard uses Streamlit. Video inference uses open-source Ultralytics YOLOv8 nano weights; weights are downloaded locally on first run. Optional summaries use Salesforce BLIP base locally and also download weights on first use. No paid AI API is required.

## 1. Project structure

```text
.
├── app.py             # Streamlit login, uploads, webcam, plans, and history
├── database.py        # SQLite users, subscriptions, quota, and incident logs
├── detector.py        # Local YOLO inference, metrics, evidence, optional BLIP
├── alerts.py          # Optional SMTP incident email with image attachment
├── requirements.txt
├── .env.example
└── README.md
```

## 2. Database and subscription logic

`database.py` creates `incident_system.db` at first app launch. It stores users with salted scrypt password hashes, subscription choices, and incident records. Basic permits 5 scans/month, Pro 100, and Enterprise 10,000. Plan selection is mock payment logic and does not charge money. The application checks monthly usage before starting each scan.

## 3. Detection and summary engine

`detector.py` scans every readable frame with YOLOv8 nano. It calculates flagged frames divided by all analyzed frames, reports the highest detector score among flagged frames, records the first and last flagged timestamps as `MM:SS`, and writes an annotated MP4 and evidence snapshot. It flags selected detectable objects (including COCO classes such as knife and baseball bat) and a basic rapid-motion/close-proximity cue. The optional BLIP captioner captions selected flagged frames locally; leave it disabled to avoid its larger model download.

The supplied YOLO COCO checkpoint is a general object detector, not a trained theft, fight, trespassing, vandalism, or weapon-recognition model. A frame percentage is a screening heuristic, not a calibrated probability or a determination that malicious activity occurred. It can miss weapons and actions and can flag benign scenes. Validate findings with a human reviewer. For a stronger dissertation evaluation, train and measure a dedicated action/object model on a suitably labeled dataset.

## 4. Alert dispatcher

`alerts.py` sends email only when the flagged-frame percentage is greater than 20% (configurable with `ALERT_THRESHOLD_PERCENT`). Configure SMTP variables in the process environment or copy `.env.example` values into your local environment. For Gmail, use an App Password and keep it private. SMTP is optional; scans and history work without it. Alerts include a summary, metrics, and the highest-scoring flagged frame when available.

## 5. Web dashboard

`app.py` provides account creation and login, video upload, a server-hosted webcam capture option, plan selection, scan metrics, annotated video playback, and incident history. The webcam option accesses camera 0 on the computer running Streamlit (useful for a local demo; it does not access a browser visitor's camera on a remote deployment). Set `APP_DATA_DIR` to a persistent private directory if you deploy the demo. The portal is a research prototype and should not be exposed publicly without production authentication, authorization, transport security, retention, and deployment controls.

## 6. Setup and execution

Use Python 3.10 or 3.11. Install a PyTorch build matching your CPU/GPU from the [official PyTorch selector](https://pytorch.org/get-started/locally/), then install the rest:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS/Linux:
# source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchvision
python -m pip install -r requirements.txt
streamlit run app.py
```

On the first scan, Ultralytics downloads `yolov8n.pt` into its local model cache. To prefetch the detector weights:

```bash
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

The optional BLIP checkpoint is fetched when the caption checkbox is selected for the first time. You can prefetch it with:

```bash
python -c "from transformers import BlipProcessor, BlipForConditionalGeneration; m='Salesforce/blip-image-captioning-base'; BlipProcessor.from_pretrained(m); BlipForConditionalGeneration.from_pretrained(m)"
```

Model weights are open source, but first-time downloads require internet access. Afterwards, set `HF_HUB_OFFLINE=1` and `YOLO_MODEL` to a local checkpoint path for offline use. Install `opencv-python` as listed (not `opencv-python-headless`) to enable webcam capture. SMTP email is not enabled unless all required environment variables are configured.
