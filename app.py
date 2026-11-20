"""Streamlit dashboard for the local malicious-activity screening demo."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st

import database as db
from alerts import ALERT_THRESHOLD, send_incident_alert
from detector import analyze_video

ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.getenv("APP_DATA_DIR", ROOT / "user_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()

st.set_page_config(page_title="Local Activity Screening", page_icon="🛡️", layout="wide")


def login_view():
    st.title("🛡️ AI-Powered Activity Screening")
    st.caption("Local open-source computer vision • SQLite demo portal")
    login_tab, signup_tab = st.tabs(["Log in", "Create account"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in", type="primary")
        if submitted:
            user = db.authenticate(email, password)
            if user:
                st.session_state.user_id = user.id
                st.session_state.email = user.email
                st.rerun()
            st.error("Email or password is incorrect.")
    with signup_tab:
        with st.form("signup_form"):
            email = st.text_input("Email address", key="signup_email")
            password = st.text_input("Password (at least 8 characters)", type="password", key="signup_password")
            submitted = st.form_submit_button("Create account", type="primary")
        if submitted:
            user, error = db.create_user(email, password)
            if error:
                st.error(error)
            else:
                st.session_state.user_id = user.id
                st.session_state.email = user.email
                st.success("Account created on the Basic plan.")
                st.rerun()


def capture_webcam(seconds: int, destination: Path) -> None:
    """Capture a short clip from the server's default camera for local demos."""
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam 0 on the machine running Streamlit.")
    fps = camera.get(cv2.CAP_PROP_FPS) or 20.0
    fps = min(max(fps, 5.0), 30.0)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        camera.release()
        raise RuntimeError("Could not create webcam recording.")
    deadline = time.monotonic() + seconds
    try:
        while time.monotonic() < deadline:
            ok, frame = camera.read()
            if ok:
                writer.write(frame)
    finally:
        camera.release()
        writer.release()


def run_scan(source: Path, source_name: str, use_caption: bool):
    user_id = st.session_state.user_id
    evidence_dir = DATA_DIR / "evidence" / str(user_id)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    token = f"scan_{int(time.time())}"
    snapshot = evidence_dir / f"{token}.jpg"
    annotated = evidence_dir / f"{token}_annotated.mp4"
    progress = st.progress(0, text="Loading local detection model…")
    try:
        result = analyze_video(
            source, output_path=annotated, snapshot_path=snapshot,
            progress_callback=lambda value: progress.progress(min(value, 1.0), text="Analyzing video locally…"),
            use_caption_model=use_caption,
        )
    finally:
        progress.empty()
    alerted, alert_message = send_incident_alert(st.session_state.email, result, str(snapshot) if snapshot.exists() else None)
    record = db.save_incident(user_id, result, source_name, str(snapshot) if snapshot.exists() else None, alerted)
    st.session_state.last_scan = {**result, "snapshot_path": str(snapshot), "annotated_path": result["video_path"], "alert_message": alert_message, "incident_id": record.id}


def dashboard():
    user_id = st.session_state.user_id
    plan, used, allowance = db.get_usage(user_id)
    with st.sidebar:
        st.write(f"Signed in as **{st.session_state.email}**")
        st.write(f"Plan: **{plan}**")
        st.caption(f"Scans this month: {used}/{allowance}")
        if st.button("Log out"):
            st.session_state.clear()
            st.rerun()
    st.title("AI-Powered Malicious Activity Detection & Alert System")
    st.caption("Local inference using open-source models. Results are screening cues and require human review.")
    tab_scan, tab_plans, tab_history = st.tabs(["Video analysis", "Subscription", "Emergency alert history"])

    with tab_scan:
        st.info("The default YOLOv8 nano weights run locally after the first download. Optional BLIP captions use a local Hugging Face model and may need several GB of RAM.")
        if used >= allowance:
            st.warning("Monthly scan limit reached. Upgrade your plan to continue.")
        upload, webcam = st.columns(2)
        uploaded = upload.file_uploader("Upload an MP4 or AVI clip", type=["mp4", "avi"], disabled=used >= allowance)
        use_caption = st.checkbox("Generate optional local image captions (downloads BLIP weights)")
        if uploaded and upload.button("Analyze uploaded video", type="primary", disabled=used >= allowance):
            if uploaded.size > 500 * 1024 * 1024:
                st.error("Maximum upload size is 500 MB.")
            else:
                with tempfile.TemporaryDirectory(prefix="local_scan_") as tmp:
                    source = Path(tmp) / Path(uploaded.name).name
                    source.write_bytes(uploaded.getbuffer())
                    try:
                        run_scan(source, uploaded.name, use_caption)
                    except Exception as exc:
                        st.exception(exc)
        webcam_seconds = webcam.slider("Webcam capture length (seconds)", 5, 30, 10)
        if webcam.button("Capture and analyze server webcam", disabled=used >= allowance):
            with tempfile.TemporaryDirectory(prefix="webcam_scan_") as tmp:
                source = Path(tmp) / "webcam.mp4"
                try:
                    with st.spinner("Recording from the webcam connected to the Streamlit host…"):
                        capture_webcam(webcam_seconds, source)
                    run_scan(source, "Live webcam", use_caption)
                except Exception as exc:
                    st.exception(exc)

        result = st.session_state.get("last_scan")
        if result:
            st.subheader("Latest screening result")
            cols = st.columns(4)
            cols[0].metric("Flagged frames", f"{result['threat_percent']:.1f}%")
            cols[1].metric("Peak confidence", f"{result['peak_confidence']:.1%}")
            cols[2].metric("Incident started", result["incident_start"] or "None")
            cols[3].metric("Incident ended", result["incident_end"] or "None")
            st.write(result["summary"])
            st.caption(f"Analyzed {result['frame_count']} frames • {result['duration_seconds']:.1f} sec • threshold {ALERT_THRESHOLD:.0f}%")
            st.info(result["alert_message"])
            if result.get("snapshot_path") and Path(result["snapshot_path"]).exists():
                st.image(result["snapshot_path"], caption="Highest-scoring flagged frame")
            if result.get("annotated_path") and Path(result["annotated_path"]).exists():
                st.video(result["annotated_path"])
            if result.get("object_counts"):
                st.write("Detector observations:", result["object_counts"])
            st.caption("The percentage is the fraction of frames flagged by a simple heuristic. It is not a calibrated probability or a finding that an offense occurred.")

    with tab_plans:
        st.subheader("Demo subscription tiers")
        columns = st.columns(3)
        for col, (name, details) in zip(columns, db.PLANS.items()):
            with col:
                st.markdown(f"### {name}")
                st.write("Free" if details["price"] == 0 else f"${details['price']}/month (mock payment)")
                st.write(f"{details['monthly_scans']} scans per month")
                if st.button(f"Select {name}", key=f"plan_{name}"):
                    db.set_plan(user_id, name)
                    st.success(f"Plan set to {name} (demo; no charge).")
                    st.rerun()
        st.caption("This dissertation demo stores plan selection locally. It does not process payments.")

    with tab_history:
        st.subheader("Emergency alert history")
        records = db.incident_history(user_id)
        if not records:
            st.write("No scans recorded yet.")
        for row in records:
            with st.expander(f"{row.created_at:%Y-%m-%d %H:%M} • {row.source_name} • {row.threat_percent:.1f}%"):
                st.write(row.summary)
                st.write(f"Start: {row.incident_start or '—'} · End: {row.incident_end or '—'} · Peak confidence: {row.peak_confidence:.1%}")
                st.write("Email alert sent" if row.alerted else "No email alert sent")
                if row.snapshot_path and Path(row.snapshot_path).exists():
                    st.image(row.snapshot_path, width=400)


if "user_id" not in st.session_state:
    login_view()
else:
    dashboard()
