import hashlib
import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

API_KEY = st.secrets["API_KEY"]

BASE_DIR = Path(__file__).resolve().parent
PROCESSOR = BASE_DIR / "main.py"

st.set_page_config(page_title="Video Activity Analysis", page_icon="🎥")
st.title("Video Activity Analysis")
st.write(
    "Upload a video to generate an annotated version. The processor analyzes "
    "up to the first 540 frames."
)
st.caption("OpenAI API key is read from Streamlit secrets.")

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
    key="video_upload",
)

if uploaded_video is None:
    st.session_state.pop("video_result", None)
else:
    upload_id = hashlib.sha256(uploaded_video.getbuffer()).hexdigest()
    previous_result = st.session_state.get("video_result")
    if previous_result and previous_result.get("upload_id") != upload_id:
        st.session_state.pop("video_result", None)

if uploaded_video and st.button("Process video", type="primary"):
    result = None
    video_bytes = None
    processor_output = ""
    upload_id = hashlib.sha256(uploaded_video.getbuffer()).hexdigest()

    try:
        if not PROCESSOR.is_file():
            raise FileNotFoundError(f"Could not find the video processor: {PROCESSOR}")

        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="video_analysis_") as temp_dir:
            input_path = Path(temp_dir) / f"uploaded_video{suffix}"
            output_path = input_path.with_name("uploaded_video_annotated.mp4")
            input_path.write_bytes(uploaded_video.getbuffer())

            process_env = os.environ.copy()
            process_env["OPENAI_API_KEY"] = API_KEY
            command = [
                sys.executable,
                str(PROCESSOR),
                str(input_path),
                "--no-preview",
            ]

            with st.spinner("Processing video. This may take a few minutes..."):
                result = subprocess.run(
                    command,
                    cwd=BASE_DIR,
                    env=process_env,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=900,
                    check=False,
                )
            processor_output = "\n".join(
                part for part in (result.stdout, result.stderr) if part
            )
            if output_path.is_file() and output_path.stat().st_size > 0:
                video_bytes = output_path.read_bytes()
    except subprocess.TimeoutExpired as exc:
        processor_output = "Processing timed out after 15 minutes."
        if exc.stdout:
            processor_output += f"\n\nPartial output:\n{exc.stdout}"
        if exc.stderr:
            processor_output += f"\n\nPartial errors:\n{exc.stderr}"
    except Exception as exc:
        processor_output = f"{type(exc).__name__}: {exc}\n\n{traceback.format_exc()}"

    st.session_state["video_result"] = {
        "video_bytes": video_bytes,
        "returncode": result.returncode if result is not None else None,
        "processor_output": processor_output,
        "file_stem": Path(uploaded_video.name).stem,
        "upload_name": uploaded_video.name,
        "upload_id": upload_id,
    }


video_result = st.session_state.get("video_result")
if video_result and uploaded_video and video_result.get("upload_id") == upload_id:
    result_video = video_result["video_bytes"]
    if result_video:
        st.success("Processed video is ready to play. Use the video controls to replay it anytime.")
        st.video(result_video, format="video/mp4")

    returncode = video_result["returncode"]
    if returncode is not None and returncode != 0:
        st.error(f"Video processing exited with code {returncode}.")
        with st.expander("Error details", expanded=True):
            st.code(video_result["processor_output"][-20000:] or "The processor returned no error text.")
    elif returncode is None or not result_video:
        st.error("Video processing failed.")
        with st.expander("Error details", expanded=True):
            st.code(video_result["processor_output"][-20000:] or "No error details were produced.")
