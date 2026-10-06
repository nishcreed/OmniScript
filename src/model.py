from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import settings

#DLA
@lru_cache(maxsize=1)
def load_layout_model() -> Any:
    """
    Load the document layout model once.
    """
    
    from ultralytics import YOLO

    model_path: Path = settings.layout_model_path

    if not model_path.exists():
        raise FileNotFoundError(
            f"Layout model was not found: {model_path}"
        )

    model = YOLO(str(model_path))

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

    processor = Blip2Processor.from_pretrained(
        "Salesforce/blip2-opt-2.7b"
    )

    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b",
        device_map="auto",
    )

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

    processor = AutoProcessor.from_pretrained(
        "Salesforce/blip2-itm-vit-g"
    )

    model = Blip2ForImageTextRetrieval.from_pretrained(
        "Salesforce/blip2-itm-vit-g",
        device_map="auto",
    )

    return model, processor


# Translation
@lru_cache(maxsize=1)
def load_translation_model() -> tuple[Any, Any]:
    """Load the English-to-Indic model and tokenizer once."""

    import torch
    from transformers import AutoModelForSeq2SeqLM
    from IndicTransTokenizer.IndicTransTokenizer import IndicTransTokenizer

    device = settings.device

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a GPU."
        )

    tokenizer = IndicTransTokenizer(direction="en-indic")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "ai4bharat/indictrans2-en-indic-dist-200M",
        trust_remote_code=True,
    )

    model = model.to(device)
    model.eval()

    return model, tokenizer
