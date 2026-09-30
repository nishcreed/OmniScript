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


#Translation
def associate_images(
    page: PageResult,
    texts: list[str],
    captions: dict[int, str],
) -> list[str]:
    """
    Place each picture caption next to its most relevant text.
    """

    model, processor = load_itm_model()

    image = Image.open(
        page.image_path
    ).convert("RGB")

    image_height = image.height

    output_texts = list(texts)

    picture_regions = page.regions.get(
        "Picture",
        [],
    )

    for index, region in enumerate(
        picture_regions,
        start=1,
    ):
        caption = captions[index]

        y1 = region.box[1]
        y2 = region.box[3]

        if y1 < 50:
            output_texts.insert(
                0,
                caption,
            )
            continue

        if y2 > image_height - 50:
            output_texts.append(
                caption,
            )
            continue

        candidates = [
            text
            for text in output_texts
            if not text.startswith(
                "A picture of "
            )
        ]

        if not candidates:
            output_texts.append(
                caption,
            )
            continue

        cropped_image = image.crop(
            tuple(region.box)
        )

        scores: list[float] = []

        for candidate in candidates:
            inputs = processor(
                images=cropped_image,
                text=candidate,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(model.device)
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                result = model(
                    **inputs,
                    use_image_text_matching_head=True,
                )

            probabilities = torch.softmax(
                result.logits_per_image,
                dim=1,
            )

            matching_probability = (
                probabilities[0][1].item()
            )

            scores.append(
                matching_probability
            )

        best_index = scores.index(
            max(scores)
        )

        best_text = candidates[best_index]

        insert_index = output_texts.index(
            best_text
        )

        output_texts.insert(
            insert_index + 1,
            caption,
        )

    return output_texts

