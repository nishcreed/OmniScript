from pathlib import Path
import tempfile

import numpy as np
import pyrubberband as pyrb
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
from io import BytesIO


def render_sentence(
    text: str,
    language: str,
    target_rate: int,
    pitch: float,
    speed: float,
) -> np.ndarray:
    """
    Convert one sentence into audio samples.
    """

    is_special = (
        text.startswith("_*")
        and text.endswith("*_.")
    )

    if is_special:
        text = text[2:-3]

    with tempfile.NamedTemporaryFile(
        suffix=".mp3"
    ) as temporary_file:
        speech = gTTS(
            text=text,
            lang=language,
        )

        speech.write_to_fp(
            temporary_file
        )

        temporary_file.flush()

        audio = AudioSegment.from_file(
            temporary_file.name,
            format="mp3",
        )

    audio = (
        audio
        .set_frame_rate(target_rate)
        .set_channels(1)
        .set_sample_width(2)
    )

    samples = (
        np.frombuffer(
            audio.raw_data,
            dtype=np.int16,
        )
        .astype(np.float32)
        / 32768.0
    )

    if not is_special:
        return samples

    pitch_shifted = pyrb.pitch_shift(
        samples,
        target_rate,
        pitch,
    )

    time_stretched = pyrb.time_stretch(
        pitch_shifted,
        target_rate,
        speed,
    )

    return time_stretched


def text_to_audio(
    texts: list[str],
    language: str,
    pitch: float = -2,
    speed: float = 1.0,
) -> bytes:
    """Generate WAV audio in memory."""

    target_rate = 22050
    gap = np.zeros(int(target_rate * 0.5), dtype=np.float32)
    segments: list[np.ndarray] = []

    for text in texts:
        if not text.strip():
            continue

        segment = render_sentence(
            text=text,
            language=language,
            target_rate=target_rate,
            pitch=pitch,
            speed=speed,
        )

        if segments:
            segments.append(gap)

        segments.append(segment)

    if not segments:
        raise ValueError("No text available for speech generation.")

    combined_audio = np.concatenate(segments)

    buffer = BytesIO()
    sf.write(
        buffer,
        combined_audio,
        target_rate,
        format="WAV",
        subtype="PCM_16",
    )

    return buffer.getvalue()