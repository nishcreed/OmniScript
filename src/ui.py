import base64
import logging
import os
from io import BytesIO
from pathlib import Path

import gradio as gr
import httpx
import soundfile as sf
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


def validate_image_upload(image_path: str | None) -> Path:
    """Validate uploads before processing, with messages intended for users."""

    if not image_path:
        raise gr.Error("Upload a PNG or JPEG document image first.")

    path = Path(image_path)

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise gr.Error(
            "Use a PNG or JPEG image. Convert HEIC images to PNG or JPEG first."
        )

    try:
        size = path.stat().st_size
        if size == 0:
            raise gr.Error("This file is empty. Choose another image.")
        if size > MAX_UPLOAD_BYTES:
            raise gr.Error("Choose an image smaller than 20 MB.")

        with Image.open(path) as image:
            if image.format not in {"PNG", "JPEG"}:
                raise gr.Error("Use a PNG or JPEG image.")
            image.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, SyntaxError):
        raise gr.Error(
            "This image cannot be read. Try another PNG or JPEG image."
        ) from None
    except OSError:
        raise gr.Error(
            "This image is unavailable or damaged. Upload it again."
        ) from None

    return path

API_BASE_URL = os.getenv(
    "OMNISCRIPT_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

LANGUAGES = [
    ("English", "en"),
    ("Tamil", "ta"),
    ("Hindi", "hi"),
    ("Telugu", "te"),
    ("Malayalam", "ml"),
    ("Kannada", "kn")
]

async def process_image(
    image_path: str | None,
    language: str,
    pitch: float,
    speed: float
):
    """Keep unexpected technical errors in logs, not in the interface."""

    try:
        return await _process_image(image_path, language, pitch, speed)
    except gr.Error:
        raise
    except Exception:
        logger.exception("Unexpected document reader failure")
        raise gr.Error(
            "Something went wrong while reading the document. Please try again."
        ) from None


async def _process_image(
    image_path: str | None,
    language: str,
    pitch: float,
    speed: float,
):
    path = validate_image_upload(image_path)

    api_key = os.getenv("RUNPOD_API_KEY")
    headers = (
        {"Authorization": f"Bearer {api_key}"}
        if api_key
        else {}
    )
    timeout = httpx.Timeout(600.0, connect=10.0)

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            trust_env=False,
            headers=headers
        ) as client:
            with path.open("rb") as image_file:
                response = await client.post(
                    f"{API_BASE_URL}/process",
                    files={
                        "image": (path.name, image_file)
                    },
                    data={
                        "language": language,
                        "pitch": str(pitch),
                        "speed": str(speed)
                    }
                )

    except httpx.TimeoutException as error:
        raise gr.Error(
            "Processing timed out. Try a smaller document."
        ) from error

    except httpx.RequestError as error:
        raise gr.Error(
            "Could not connect to the document processing API."
        ) from error

    if response.is_error:
        if response.status_code >= 500:
            logger.error("Processing API returned HTTP %s", response.status_code)
            raise gr.Error(
                "Could not read this document. Please try again."
            )

        if response.status_code == 413:
            raise gr.Error("Choose an image smaller than 20 MB.")

        if response.status_code == 422:
            raise gr.Error("Check the selected language, pitch, and speed.")

        try:
            detail = response.json().get("detail")
        except (ValueError, AttributeError):
            detail = None

        # Forward only recognized messages, never arbitrary backend details.
        messages = {
            "Cannot read the uploaded image.":
                "This image cannot be read. Try another PNG or JPEG image.",
            "No readable text or picture descriptions were found.":
                "No readable content was found. Try a clearer document image.",
            "Translation produced empty narration.":
                "Could not translate this document. Try another language.",
        }

        message = messages.get(detail) if isinstance(detail, str) else None
        raise gr.Error(message or "Could not read this document. Try another image.")

    try:
        payload = response.json()

        audio = payload["audio"]

        if(
            audio["encoding"] != "base64" or
            audio["mime_type"] != "audio/wav"
        ):
            raise ValueError("Unexpected audio format.")

        wav_bytes = base64.b64decode(
            audio["data"],
            validate=True
        )

        samples, sample_rate = sf.read(
            BytesIO(wav_bytes),
            dtype="float32"
        )

        # Hide narration markers in the displayed text.
        lines = [
            text[2:-3]
            if text.startswith("_*") and text.endswith("*_.")
            else text
            for text in payload["text"]
        ]

    except (KeyError, TypeError, ValueError, RuntimeError) as error:
        logger.exception("Invalid processing API response")
        raise gr.Error(
            "Could not load the result. Please try again."
        ) from None

    return "\n\n".join(lines), (sample_rate, samples)


def build_ui() -> gr.Blocks:
    """Build the document reader interface."""

    with gr.Blocks(
        title="OmniScript",
        analytics_enabled=False,
        delete_cache=(600, 3600),
    ) as demo:
        gr.Markdown(
            "# OmniScript\n"
            "Upload a document image to read and listen to its content."
        )

        with gr.Row():
            with gr.Column():
                image = gr.File(
                    label="Document image (PNG or JPEG, up to 20 MB)",
                    file_types=[".png", ".jpg", ".jpeg"],
                    file_count="single",
                    type="filepath",
                )

                language = gr.Dropdown(
                    choices=LANGUAGES,
                    value="en",
                    label="Output language",
                )

                pitch = gr.Slider(
                    minimum=-12,
                    maximum=12,
                    value=-2,
                    step=1,
                    label="Pitch",
                )

                speed = gr.Slider(
                    minimum=0.5,
                    maximum=2.0,
                    value=1.0,
                    step=0.1,
                    label="Speed",
                )

                gr.Markdown(
                    "Pitch and speed apply to headings and descriptions."
                )

                process_button = gr.Button(
                    "Read document",
                    variant="primary",
                )

            with gr.Column():
                text_output = gr.Textbox(
                    label="Document narration",
                    lines=15,
                    interactive=False,
                )

                audio_output = gr.Audio(
                    label="Listen",
                    interactive=False,
                    format="wav",
                )

        process_button.click(
            fn=process_image,
            inputs=[image, language, pitch, speed],
            outputs=[text_output, audio_output],
            concurrency_limit=1,
            show_progress="full",
        )

    demo.queue(max_size=10)
    return demo
