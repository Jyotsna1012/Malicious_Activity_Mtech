"""Streamlit UI for local, quantized fight screening."""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")
from detector import ModelUnavailableError, analyze_video

st.set_page_config(page_title="Video Activity Screening", page_icon="🎥", layout="wide")
st.title("AI-Powered Malicious Activity Detection")
st.caption("Local CPU screening with a compact temporal classifier. Treat every flag as a review cue.")

upload = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
if upload is not None and st.button("Analyze video", type="primary"):
    st.session_state.pop("screening_result", None)
    if upload.size > 200 * 1024 * 1024:
        st.error("For this low-memory deployment, upload videos smaller than 200 MB.")
        st.stop()
    progress = st.progress(0.0, text="Preparing local classifier")

    def report(value: float, message: str) -> None:
        progress.progress(min(1.0, max(0.0, float(value))), text=message)

    try:
        suffix = Path(upload.name).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="screening_") as directory:
            source = Path(directory) / f"source{suffix}"
            annotated = Path(directory) / "annotated.mp4"
            storyboard = Path(directory) / "storyboard.jpg"
            source.write_bytes(upload.getbuffer())
            result = analyze_video(source, output_path=annotated, snapshot_path=storyboard,
                                   progress_callback=report)
            result["annotated_bytes"] = annotated.read_bytes() if annotated.is_file() else None
            result["storyboard_bytes"] = storyboard.read_bytes() if storyboard.is_file() else None
        st.session_state["screening_result"] = result
    except ModelUnavailableError as exc:
        st.error(str(exc))
        st.info("Training instructions and the required clip labels are in README.md under ‘Train the compact fight model’.")
    except Exception as exc:
        st.error(f"Video analysis failed: {type(exc).__name__}: {exc}")
    finally:
        progress.empty()

result = st.session_state.get("screening_result")
if result:
    st.subheader("Screening result")
    columns = st.columns(4)
    columns[0].metric("Possible fight", "Flagged" if result["possible_fight"] else "Not flagged")
    columns[1].metric("Flagged frames", f"{result['malicious_frames']}/{result['frame_count']}")
    columns[2].metric("Flagged video", f"{result['threat_percent']:.1f}%")
    columns[3].metric("Peak model score", f"{result['peak_confidence']:.1%}")
    if result["possible_fight"] or result.get("no_candidate_windows"):
        st.warning(result["summary"])
    else:
        st.success(result["summary"])
    st.caption(f"Model: {result['action_model']} · Runtime: {result['device']}")
    st.caption(f"Motion candidate windows: {result['candidate_window_count']}/{result['window_count']}")
    if result["incident_runs"]:
        st.markdown("**Incident windows**")
        for incident in result["incident_runs"]:
            st.write(f"{incident['start']}–{incident['end']} · score {incident['confidence']:.1%}")
    if result.get("storyboard_bytes"):
        st.image(result["storyboard_bytes"], caption="Four-frame incident evidence", width="stretch")
    if result.get("annotated_bytes"):
        st.video(result["annotated_bytes"])
    st.caption("Automated screening can miss events or flag benign motion. Review the original video.")
