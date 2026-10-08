"""Orchestrate the document-reading inference stages."""

from pathlib import Path

from PIL import Image

from src.data import ReaderResult

from .audio import text_to_audio
from .captioning import caption_images
from .layout import detect_layout
from .matching import associate_images
from .ocr import extract_text
from .translation import translate_text


def process_document(
    image_path: Path,
    language: str = "en",
    pitch: float = -2,
    speed: float = 1.0,
) -> ReaderResult:
    """Run layout, OCR, captioning, matching, translation, and TTS."""

    import math
    from PIL import UnidentifiedImageError

    image_path = Path(image_path)
    language = language.strip().lower()

    if language not in {"en", "ta", "hi", "te", "ml", "kn"}:
        raise ValueError(f"Unsupported language: {language}")

    if not math.isfinite(pitch) or not -12 <= pitch <= 12:
        raise ValueError("Pitch must be between -12 and 12 semitones.")

    if not math.isfinite(speed) or not 0.5 <= speed <= 2.0:
        raise ValueError("Speed must be between 0.5 and 2.0.")

    if not image_path.is_file():
        raise ValueError(f"Image file does not exist: {image_path}")

    try:
        with Image.open(image_path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("Cannot read the uploaded image.") from error

    page = detect_layout(image_path)
    texts = extract_text(page)

    captions = (
        caption_images(page)
        if page.regions.get("Picture")
        else {}
    )

    # Also splits ordinary text into sentences when there are no pictures.
    texts = associate_images(
        page=page,
        texts=texts,
        captions=captions,
    )

    if not texts:
        raise ValueError(
            "No readable text or picture descriptions were found."
        )

    translated_texts = translate_text(
        texts=texts,
        language=language,
    )

    # Reject empty narration instead of generating a silent WAV.
    if not translated_texts or any(
        not text.strip()
        or (
            text.startswith("_*")
            and text.endswith("*_.")
            and not text[2:-3].strip()
        )
        for text in translated_texts
    ):
        raise ValueError("Translation produced empty narration.")

    audio_bytes = text_to_audio(
        texts=translated_texts,
        language=language,
        pitch=pitch,
        speed=speed,
    )

    page.text = translated_texts

    return ReaderResult(
        text=translated_texts,
        pages=[page],
        audio_bytes=audio_bytes,
    )
