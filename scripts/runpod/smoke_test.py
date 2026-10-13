"""GPU model-loading and document-inference smoke checks."""

import argparse
from io import BytesIO
from pathlib import Path
import subprocess
import sys
import time


MODEL_NAMES = ("layout", "captioning", "matching", "translation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    load = commands.add_parser("load", help="Check model loading")
    load.add_argument("model", choices=(*MODEL_NAMES, "all"))

    infer = commands.add_parser("infer", help="Check document inference")
    infer.add_argument("image", type=Path)
    infer.add_argument(
        "--language", choices=("en", "ta", "hi", "te", "ml", "kn"), default="en"
    )
    return parser.parse_args()


def check_inference(image: Path, language: str):
    import numpy as np
    import soundfile as sf

    from src.inference.pipeline import process_document

    result = process_document(image, language=language, pitch=0, speed=1.0)
    if not result.text or not all(text.strip() for text in result.text):
        raise RuntimeError("Empty narration.")

    audio, sample_rate = sf.read(BytesIO(result.audio_bytes))
    if audio.size == 0 or not np.isfinite(audio).all():
        raise RuntimeError("Empty or invalid audio.")
    if not np.any(audio):
        raise RuntimeError("Silent audio.")

    print("Narration:")
    print("\n".join(result.text))
    print(f"Audio duration: {len(audio) / sample_rate:.1f} seconds")
    return result


def main() -> None:
    args = parse_args()

    if args.command == "load" and args.model == "all":
        # Separate processes release each model before loading the next one.
        for name in MODEL_NAMES:
            subprocess.run(
                [sys.executable, "-m", "scripts.runpod.smoke_test", "load", name],
                check=True,
            )
        print("All model loading checks passed.")
        return

    if args.command == "infer" and not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")

    import torch

    from src.model import (
        load_captioning_model,
        load_itm_model,
        load_layout_model,
        load_translation_model,
        resolve_device,
    )

    loaders = dict(zip(MODEL_NAMES, (
        load_layout_model, load_captioning_model,
        load_itm_model, load_translation_model,
    )))
    device = resolve_device()
    if device.type != "cuda":
        raise RuntimeError("CUDA device required.")

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    if args.command == "load":
        print(f"Loading {args.model} on {device}...", flush=True)
        # Retain the model until after memory measurements.
        result = loaders[args.model]()
    else:
        print(f"Processing {args.image} on {device}...", flush=True)
        result = check_inference(args.image, args.language)

    torch.cuda.synchronize(device)
    print(f"Check passed in {time.perf_counter() - started:.1f} seconds")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GiB")
    print(f"Peak GPU memory allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GiB")


if __name__ == "__main__":
    main()
