"""Local API/UI smoke testing without downloading or loading models."""

import math
import struct
import wave
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.data import PageResult, ReaderResult, Region


def process_document(
    image_path: Path,
    language: str = "en",
    pitch: float = -2,
    speed: float = 1.0,
) -> ReaderResult:
    """Return clearly marked sample text and a short tone, not real speech."""

    image_path = Path(image_path)
    language = language.strip().lower()

    if language not in {"en", "ta", "hi", "te", "ml", "kn"}:
        raise ValueError(f"Unsupported language: {language}")
    if not math.isfinite(pitch) or not -12 <= pitch <= 12:
        raise ValueError("Pitch must be between -12 and 12 semitones.")
    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ValueError("Speed must be between 0.5 and 2.0.")

    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("Cannot read the uploaded image.") from error

    texts = [
        "Mock mode: this is sample narration, not extracted document text.",
        f"Image: {width} x {height}. Selected language: {language}.",
        f"Pitch: {pitch}; speed: {speed}. The audio is a test tone.",
    ]

    sample_rate = 22050
    frequency = 440 * 2 ** (pitch / 12)
    sample_count = int(sample_rate / speed)
    samples = bytearray()

    for index in range(sample_count):
        # Fade the edges to avoid clicks during playback.
        envelope = min(1.0, index / 220, (sample_count - 1 - index) / 220)
        amplitude = 0.2 * envelope * math.sin(
            2 * math.pi * frequency * index / sample_rate
        )
        samples.extend(struct.pack("<h", int(amplitude * 32767)))

    buffer = BytesIO()
    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples)

    page = PageResult(
        image_path=image_path,
        regions={"Text": [Region("Text", [0, 0, width, height], 1.0)]},
        text=texts,
    )
    return ReaderResult(text=texts, pages=[page], audio_bytes=buffer.getvalue())
