"""Streamlit customer portal for local temporal activity screening."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

import database as db
from alerts import send_incident_alert
from detector import analyze_video

DATA_DIR = Path(os.getenv("APP_DATA_DIR", ROOT / "user_data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)
db.init_db()

st.set_page_config(page_title="Local Activity Screening", page_icon="🛡️", layout="wide")


def run_scan(source: Path | int, source_name: str, *, live_seconds: int | None = None,
             enable_vlm: bool = False) -> None:
    user_id = st.session_state.user_id
    db.assert_quota_available(user_id)
    evidence_dir = DATA_DIR / "evidence" / str(user_id)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    token = f"scan_{int(time.time())}"
    storyboard = evidence_dir / f"{token}_storyboard.jpg"
    annotated = evidence_dir / f"{token}_annotated.mp4"
    progress = st.progress(0, text="Loading local pose and action models…")
    live_preview = st.empty()
    live_status = st.empty()

    def update_progress(value: float, message: str = "Analyzing locally") -> None:
        progress.progress(min(max(value, 0.0), 1.0), text=message)

    def update_preview(frame, timestamp: float, score: float, status: str) -> None:
        live_preview.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB",
                           caption=f"Live pose screen at {timestamp:.1f}s")
        live_status.info(f"{status} · kinematic cue {score:.0%} · {timestamp:.1f}s")

    try:
        result = analyze_video(
            source, output_path=annotated, snapshot_path=storyboard,
            progress_callback=update_progress, live_seconds=live_seconds,
            frame_callback=update_preview if isinstance(source, int) else None,
            use_caption_model=enable_vlm, enable_vlm=enable_vlm,
        )
    finally:
        progress.empty()
        live_status.empty()
    threshold = db.get_alert_threshold(user_id)
    evidence_path = result.get("storyboard_path")
    alerted, alert_message = send_incident_alert(
        os.getenv("ALERT_TO", ""), result, evidence_path, threshold=threshold
    )
    record = db.save_incident(user_id, result, source_name, evidence_path, alerted)
    st.session_state.last_scan = {
        **result, "snapshot_path": evidence_path, "annotated_path": result["video_path"],
        "alert_message": alert_message, "incident_id": record.id,
    }


def show_result(result: dict) -> None:
    st.subheader("Latest screening result")
    columns = st.columns(5)
    columns[0].metric("Malicious frames", f"{result['malicious_frames']}/{result['frame_count']}")
    columns[1].metric("Threat percentage", f"{result['threat_percent']:.1f}%")
    columns[2].metric("Peak threat score", f"{result['peak_confidence']:.1%}")
    columns[3].metric("Incident start", result["incident_start"] or "None")
    columns[4].metric("Incident end", result["incident_end"] or "None")
    if result.get("possible_fight"):
        st.warning(result["summary"], icon="⚠️")
    elif result["threat_percent"]:
        st.info(result["summary"])
    else:
        st.success(result["summary"])
    st.caption(f"Action model: {result['action_model']} · Inference device: {result['device']}")
    st.caption(f"Email alert status: {result['alert_message']}")
    if result.get("action_labels"):
        st.write("Detected temporal action labels:", result["action_labels"])
    elif result.get("top_action_labels"):
        st.write("Candidate-window action descriptions:", result["top_action_labels"])
    if result.get("object_counts"):
        st.write("Custom object detector labels:", result["object_counts"])
    if result.get("incident_runs"):
        st.markdown("**Incident timeline**")
        for run in result["incident_runs"]:
            st.write(f"{run['start']}–{run['end']} ({run['start_seconds']:.2f}s–{run['end_seconds']:.2f}s)")
    if result.get("snapshot_path") and Path(result["snapshot_path"]).is_file():
        st.image(result["snapshot_path"], caption="Four-frame storyboard evidence", width="stretch")
    if result.get("annotated_path") and Path(result["annotated_path"]).is_file():
        st.video(result["annotated_path"])
    st.caption("Screening cues can be wrong or incomplete. Review the source video and evidence; this does not establish a crime or intent.")


def analyzer_page(user_id: int, plan: str, used: int, allowance: int | None) -> None:
    st.header("Threat analyzer")
    st.caption("Pose kinematics find candidate interactions; VideoMAE evaluates 16-frame windows. Qwen2-VL can optionally summarize the storyboard when sufficient memory is available.")
    if allowance is not None and used >= allowance:
        st.warning("Monthly scan limit reached. Choose another tier to continue.")
    disabled = allowance is not None and used >= allowance
    uploaded = st.file_uploader("Upload MP4 or AVI video", type=["mp4", "avi"], disabled=disabled)
    enable_vlm = st.checkbox("Generate a local Qwen2-VL storyboard summary", value=False,
                             help="This optional 2B model needs substantial free RAM. It is off by default to keep hosted deployments responsive.")
    if uploaded and st.button("Analyze uploaded video", type="primary", disabled=disabled, key="analyze_upload"):
        if uploaded.size > 500*1024*1024:
            st.error("Maximum upload size is 500 MB.")
        else:
            try:
                with tempfile.TemporaryDirectory(prefix="activity_scan_") as temp_dir:
                    source = Path(temp_dir) / Path(uploaded.name).name
                    source.write_bytes(uploaded.getbuffer())
                    run_scan(source, uploaded.name, enable_vlm=enable_vlm)
            except Exception as exc:
                st.error(f"Video analysis failed: {exc}")
    if db.PLANS[plan]["live_webcam"]:
        duration = st.slider("Server webcam analysis duration (seconds)", 10, 120, 30, 5)
        if st.button("Start live webcam analysis", disabled=disabled, key="live_webcam"):
            try:
                with st.spinner("Opening the webcam attached to the Streamlit host…"):
                    run_scan(0, "Live server webcam", live_seconds=duration, enable_vlm=enable_vlm)
            except Exception as exc:
                st.error(f"Webcam analysis failed: {exc}")
    else:
        st.caption("Live webcam analysis is available on Pro and Enterprise plans.")
    latest = st.session_state.get("last_scan")
    if latest:
        show_result(latest)


def subscription_page(user_id: int, plan: str, used: int, allowance: int | None) -> None:
    st.header("Subscription plans")
    st.caption("Plan upgrades use a simulated checkout. No payment is taken.")
    columns = st.columns(3)
    for column, (name, details) in zip(columns, db.PLANS.items()):
        with column:
            st.markdown(f"### {name}")
            st.write("Free" if details["price"] == 0 else f"${details['price']}/month (mock checkout)")
            st.write("Unlimited scans" if details["monthly_scans"] is None else f"{details['monthly_scans']} scans/month")
            st.write("Live webcam: " + ("Included" if details["live_webcam"] else "Not included"))
            st.write("Custom alert threshold: " + ("Included" if details["custom_threshold"] else "20% default"))
            if st.button("Current plan" if plan == name else f"Upgrade to {name}", key=f"plan_{name}", disabled=plan == name):
                db.set_plan(user_id, name)
                st.success(f"Plan updated to {name} (simulated checkout).")
                st.rerun()
    st.caption(f"Usage this month: {used}/{'Unlimited' if allowance is None else allowance}")
    if plan == "Enterprise":
        threshold = db.get_alert_threshold(user_id)
        with st.form("enterprise_threshold"):
            value = st.slider("Email alert threshold (%)", 1.0, 100.0, float(threshold), 1.0)
            if st.form_submit_button("Save alert threshold"):
                db.set_alert_threshold(user_id, value)
                st.success("Alert threshold saved.")


def history_page(user_id: int) -> None:
    st.header("Incident history")
    records = db.incident_history(user_id)
    if not records:
        st.info("No scans recorded yet.")
        return
    for row in records:
        title = f"{row.created_at:%Y-%m-%d %H:%M} · {row.source_name} · {row.threat_percent:.1f}%"
        with st.expander(title):
            st.write(row.summary)
            st.write(f"Peak score: {row.peak_confidence:.1%} · Email alert sent: {'Yes' if row.alerted else 'No'}")
            if row.action_labels:
                st.write("Action labels:", row.action_labels)
            if row.top_action_labels:
                st.write("Candidate-window descriptions:", row.top_action_labels)
            for run in row.incident_runs:
                st.write(f"Incident: {run['start']}–{run['end']}")
            evidence = row.storyboard_path or row.snapshot_path
            if evidence and Path(evidence).is_file():
                st.image(evidence, caption="Incident storyboard", width=700)


def dashboard() -> None:
    user_id = st.session_state.user_id
    plan, used, allowance = db.get_usage(user_id)
    with st.sidebar:
        st.write("Shared workspace")
        st.write(f"Plan: **{plan}**")
        st.caption(f"Scans this month: {used}/{'Unlimited' if allowance is None else allowance}")
        page = st.radio("Portal", ["Threat analyzer", "Subscription", "Incident history"])
    st.title("AI-Powered Malicious Activity Detection & Alert System")
    st.caption("Local inference only. No paid API keys are used.")
    if page == "Threat analyzer":
        analyzer_page(user_id, plan, used, allowance)
    elif page == "Subscription":
        subscription_page(user_id, plan, used, allowance)
    else:
        history_page(user_id)


if "user_id" not in st.session_state:
    workspace_user = db.get_workspace_user()
    st.session_state.user_id = workspace_user.id
dashboard()
