import os
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PROCESSOR = BASE_DIR / "main.py"

st.set_page_config(page_title="Video Activity Analysis", page_icon="🎥")
st.title("Video Activity Analysis")
st.write(
    "Upload a video to generate an annotated version. The processor analyzes "
    "up to the first 20 seconds."
)
config_path = Path(os.environ.get("APP_CONFIG_FILE", BASE_DIR / "config.json"))
st.caption(f"OpenAI API key is read from the server config file: {config_path}")
if not config_path.is_file():
    st.info(
        "On this machine, create config.json by copying config.example.json, "
        "then replace the sample value with your API key."
    )

uploaded_video = st.file_uploader(
    "Choose a video file",
    type=["mp4", "avi", "mov", "mkv", "webm"],
)

if uploaded_video and st.button("Process video", type="primary"):
    result = None
    video_bytes = None
    processor_output = ""

    try:
        if not PROCESSOR.is_file():
            raise FileNotFoundError(f"Could not find the video processor: {PROCESSOR}")
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config file not found: {config_path}. Create it from config.example.json."
            )

        suffix = Path(uploaded_video.name).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="video_analysis_") as temp_dir:
            input_path = Path(temp_dir) / f"uploaded_video{suffix}"
            output_path = input_path.with_name("uploaded_video_annotated.mp4")
            input_path.write_bytes(uploaded_video.getbuffer())

            process_env = os.environ.copy()
            process_env["APP_CONFIG_FILE"] = str(config_path)
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

    if video_bytes:
        st.success("Processed video is ready to preview and download.")
        st.video(video_bytes)
        st.download_button(
            "Download processed video",
            data=video_bytes,
            file_name=f"{Path(uploaded_video.name).stem}_annotated.mp4",
            mime="video/mp4",
            type="primary",
        )

    if result is not None and result.returncode != 0:
        st.error(f"Video processing exited with code {result.returncode}.")
        with st.expander("Error details", expanded=True):
            st.code(processor_output[-20000:] or "The processor returned no error text.")
    elif result is None or not video_bytes:
        st.error("Video processing failed.")
        with st.expander("Error details", expanded=True):
            st.code(processor_output[-20000:] or "No error details were produced.")
