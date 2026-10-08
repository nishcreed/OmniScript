"""Printed-caption extraction and generated picture descriptions."""

from PIL import Image
import torch

from src.config import settings
from src.data import PageResult
from src.model import load_captioning_model
from .ocr import clean_text, preprocess_image


def caption_images(page: PageResult) -> dict[int, str]:
    """Read nearby printed captions, or generate picture descriptions."""

    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_command

    captions: dict[int, str] = {}
    printed_captions = page.regions.get("Caption", [])

    with Image.open(page.image_path) as source:
        image = source.convert("RGB")

        # Preserve this order: associate_images() uses the same indices.
        for index, picture in enumerate(
            page.regions.get("Picture", []),
            start=1,
        ):
            px1, py1, px2, py2 = picture.box

            # Look below the picture, with horizontal overlap.
            nearby = [
                region
                for region in printed_captions
                if abs(region.box[1] - py2) <= 30
                and min(px2, region.box[2]) > max(px1, region.box[0])
            ]
            nearby.sort(key=lambda region: abs(region.box[1] - py2))

            printed_text = ""

            for region in nearby:
                crop = image.crop(tuple(region.box))
                printed_text = clean_text(
                    pytesseract.image_to_string(preprocess_image(crop))
                )

                if printed_text:
                    break

            if printed_text:
                captions[index] = f"_*A picture: {printed_text}*_."
                continue

            # Load the cached model only when generation is needed.
            model, processor = load_captioning_model()

            crop = image.crop(tuple(picture.box))

            inputs = processor(
                images=crop,
                text="",
                return_tensors="pt",
            )

            # Cast image tensors, keeping token IDs as integers.
            inputs = {
                key: (
                    value.to(device=model.device, dtype=model.dtype)
                    if value.is_floating_point()
                    else value.to(model.device)
                )
                for key, value in inputs.items()
            }

            with torch.inference_mode():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=64,
                )

            description = processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0].strip()

            captions[index] = (
                f"_*A picture of {description}*_."
                if description
                else "_*A picture*_."
            )

    return captions
