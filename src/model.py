from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import settings

def resolve_device() -> Any:
    import torch

    device_name = settings.device

    if device_name == "auto":
        device_name = (
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device_name)

    if device.type not in {"cpu", "cuda"}:
        raise ValueError(
            "Device must be auto, cpu, cuda, or cuda:N."
        )

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but no GPU is available."
            )

        index = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )

        if index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device {index} does not exist.")

        device = torch.device(f"cuda:{index}")

    return device

def resolve_dtype(device: Any) -> Any:
    import torch

    allowed = {"auto", "float16", "bfloat16", "float32"}

    if settings.dtype not in allowed:
        raise ValueError(
            "Dtype must be auto, float16, bfloat16, or float32."
        )

    # Use full precision for our CPU inference path.
    if device.type == "cpu":
        return torch.float32

    with torch.cuda.device(device):
        supports_bf16 = torch.cuda.is_bf16_supported()

    if settings.dtype == "auto":
        return (
            torch.bfloat16 if supports_bf16 else torch.float16
        )

    if settings.dtype == "bfloat16" and not supports_bf16:
        raise ValueError(
            "This GPU does not support BF16. Use float16 or auto."
        )

    return getattr(torch, settings.dtype)
    

#DLA
@lru_cache(maxsize=1)
def load_layout_model() -> Any:
    """Load and cache the layout model on the selected device."""

    from ultralytics import YOLO

    model_path = settings.layout_model_path

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Layout model was not found: {model_path}"
        )

    device = resolve_device()

    model = YOLO(str(model_path))
    model = model.to(device)

    return model


#Caption Generation
@lru_cache(maxsize=1)
def load_captioning_model() -> tuple[Any, Any]:
    """
    Load the BLIP-2 captioning model and processor once.
    """

    from transformers import (
        Blip2ForConditionalGeneration,
        Blip2Processor,
    )

    model_name = "Salesforce/blip2-opt-2.7b"

    device = resolve_device()
    dtype = resolve_dtype(device)

    processor = Blip2Processor.from_pretrained(model_name)

    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
    )

    model = model.to(device)
    model.eval()

    return model, processor


#Image-Text Matching
@lru_cache(maxsize=1)
def load_itm_model() -> tuple[Any, Any]:
    """
    Load the BLIP-2 image-text matching model once.
    """

    from transformers import (
        AutoProcessor,
        Blip2ForImageTextRetrieval,
    )

    model_name = "Salesforce/blip2-itm-vit-g"

    device = resolve_device()
    dtype = resolve_dtype(device)

    processor = AutoProcessor.from_pretrained(model_name)

    model = Blip2ForImageTextRetrieval.from_pretrained(
        model_name,
        torch_dtype=dtype
    )

    model = model.to(device)
    model.eval()

    return model, processor


# Translation
@lru_cache(maxsize=1)
def load_translation_model() -> tuple[Any, Any]:
    """Load the English-to-Indic model and tokenizer once."""

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = "ai4bharat/indictrans2-en-indic-dist-200M"

    device = resolve_device()
    dtype = resolve_dtype(device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation="eager"
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer
