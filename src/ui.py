import base64
import os
from io import BytesIO
from pathlib import Path

import gradio as gr
import httpx
import soundfile as sf

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
    if not image_path:
        raise gr.Error("Upload a document image first.")

    path = Path(image_path)

    timeout = httpx.Timeout(600.0, connect=10.0)

    try:
        async with httpx.AsyncClient(
            timeout=timeout,
            trust_env=False,
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
            raise gr.Error(
                "Document processing failed. Check the server logs."
            )

        try:
            detail = response.json().get("detail", "Invalid request.")
        except ValueError:
            detail = "Invalid request."

        raise gr.Error(str(detail))

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
        raise gr.Error(
            "The API returned an invalid text or audio response."
        ) from error

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
                image = gr.Image(
                    label="Document image",
                    sources=["upload"],
                    type="filepath",
                    format="png",
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

