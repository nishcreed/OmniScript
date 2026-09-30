from collections.abc import Iterable

from src.data import (
    PageResult,
    ReaderResult,
    Region,
)
from src.config import settings
from src.model import (
    load_captioning_model,
    load_itm_model,
    load_layout_model,
    load_translation_model,
)
from src.audio import text_to_audio

from pathlib import Path
from PIL import Image
import torch



#DLA
def calculate_iou(
    box1: list[float],
    box2: list[float],
) -> float:
    """
    Calculate the intersection-over-union of two boxes.

    Each box uses this format:

        [x1, y1, x2, y2]
    """

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width * intersection_height
    )

    box1_area = (
        max(0, box1[2] - box1[0])
        * max(0, box1[3] - box1[1])
    )

    box2_area = (
        max(0, box2[2] - box2[0])
        * max(0, box2[3] - box2[1])
    )

    union_area = (
        box1_area
        + box2_area
        - intersection_area
    )

    if union_area == 0:
        return 0.0

    return intersection_area / union_area


def resolve_overlap(
    regions: Iterable[Region],
    iou_threshold: float = 0.5,
) -> list[Region]:
    """
    Remove overlapping regions.

    If two regions overlap too much, keep the one with
    the higher confidence score.
    """

    kept_regions: list[Region] = []

    for region in regions:
        overlapping_index = None

        for index, kept_region in enumerate(kept_regions):
            iou = calculate_iou(
                region.box,
                kept_region.box,
            )

            if iou > iou_threshold:
                overlapping_index = index
                break

        if overlapping_index is None:
            kept_regions.append(region)
            continue

        existing_region = kept_regions[overlapping_index]

        if region.score > existing_region.score:
            kept_regions[overlapping_index] = region

    return kept_regions


def sort_regions(
    regions: Iterable[Region],
    y_threshold: int = 10,
) -> list[Region]:
    """
    Sort regions from top to bottom and left to right.

    Regions whose top edges are within y_threshold pixels
    are treated as belonging to the same horizontal line.
    """

    def sort_key(region: Region) -> tuple[int, float]:
        x_coordinate = region.box[0]
        y_coordinate = region.box[1]

        y_group = int(y_coordinate // y_threshold)

        return y_group, x_coordinate

    return sorted(
        regions,
        key=sort_key,
    )


def detect_layout(image_path: Path) -> PageResult:
    """
    Detect document regions in one image using YOLO.
    """

    model = load_layout_model()

    results = model(
        source=str(image_path),
        verbose=False,
    )

    detected_regions: list[Region] = []

    for result in results:
        for box in result.boxes:
            values = (
                box.data[0]
                .detach()
                .cpu()
                .tolist()
            )

            coordinates = [
                values[0] - 2,
                values[1] - 2,
                values[2] + 2,
                values[3] + 2,
            ]

            score = float(values[4])
            class_id = int(values[5])
            label = result.names[class_id]

            region = Region(
                label=label,
                box=coordinates,
                score=score,
            )

            detected_regions.append(region)

    cleaned_regions = resolve_overlap(
        detected_regions
    )

    ordered_regions = sort_regions(
        cleaned_regions
    )

    grouped_regions: dict[str, list[Region]] = {}

    for region in ordered_regions:
        if region.label not in grouped_regions:
            grouped_regions[region.label] = []

        grouped_regions[region.label].append(region)

    return PageResult(
        image_path=image_path,
        regions=grouped_regions,
    )


# OCR
def preprocess_image(
    image: Image.Image,
    resize_factor: int = 4,
) -> Image.Image:
    """
    Prepare a cropped image for OCR.
    """

    image = image.convert("RGB")

    new_width = image.width * resize_factor
    new_height = image.height * resize_factor

    resized_image = image.resize(
        (new_width, new_height),
        Image.Resampling.BICUBIC,
    )

    return resized_image


def clean_text(text: str) -> str:
    """
    Remove empty lines and extra whitespace from OCR output.
    """

    lines = text.splitlines()

    cleaned_lines = [
        line.strip()
        for line in lines
        if line.strip()
    ]

    return " ".join(cleaned_lines)


def extract_text(page: PageResult) -> list[str]:
    """
    Extract text from all Text regions on one page.
    """

    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = (
        settings.tesseract_command
    )

    image = Image.open(page.image_path)

    text_regions = page.regions.get(
        "Text",
        [],
    )

    ordered_regions = sort_regions(
        text_regions
    )

    extracted_text: list[str] = []

    for region in ordered_regions:
        cropped_image = image.crop(
            tuple(region.box)
        )

        processed_image = preprocess_image(
            cropped_image
        )

        raw_text = pytesseract.image_to_string(
            processed_image
        )

        text = clean_text(
            raw_text
        )

        if text:
            extracted_text.append(text)

    page.text = extracted_text

    return extracted_text


#Caption Generation
def caption_images(page: PageResult) -> dict[int, str]:
    """
    Generate captions for all Picture regions on a page.
    """

    model, processor = load_captioning_model()

    image = Image.open(
        page.image_path
    ).convert("RGB")

    captions: dict[int, str] = {}

    picture_regions = page.regions.get(
        "Picture",
        [],
    )

    for index, region in enumerate(
        picture_regions,
        start=1,
    ):
        cropped_image = image.crop(
            tuple(region.box)
        )

        inputs = processor(
            images=cropped_image,
            text="",
            return_tensors="pt",
        )

        inputs = {
            key: value.to(model.device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs
            )

        caption = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0].strip()

        captions[index] = (
            f"A picture of {caption}."
        )

    return captions


#Image-Text Matching
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


#Translation
def translate_text(
    texts: list[str],
    language: str,
    batch_size: int = 8,
) -> list[str]:
    """
    Translate English text into an Indic language.
    """

    language_codes = {
        "ta": "tam_Taml",
        "hi": "hin_Deva",
        "te": "tel_Telu",
        "ml": "mal_Mlym",
        "kn": "kan_Knda",
    }

    if language == "en":
        return texts

    if language not in language_codes:
        raise ValueError(
            f"Unsupported language: {language}"
        )

    import torch

    from IndicTransTokenizer.IndicTransTokenizer import (
        IndicProcessor,
    )

    model, tokenizer = load_translation_model()

    target_language = language_codes[language]

    processor = IndicProcessor(
        inference=True
    )

    special_flags = [
        text.startswith("_*")
        and text.endswith("*_.")
        for text in texts
    ]

    cleaned_texts = [
        text[2:-3] if is_special else text
        for text, is_special in zip(
            texts,
            special_flags,
        )
    ]

    device = next(
        model.parameters()
    ).device

    translated_texts: list[str] = []

    for start in range(
        0,
        len(cleaned_texts),
        batch_size,
    ):
        batch_texts = cleaned_texts[
            start:start + batch_size
        ]

        batch_flags = special_flags[
            start:start + batch_size
        ]

        prepared_batch = processor.preprocess_batch(
            batch_texts,
            src_lang="eng_Latn",
            tgt_lang=target_language,
        )

        tokenized_batch = tokenizer(
            prepared_batch,
            src=True,
            return_tensors="pt",
            padding="longest",
        )

        tokenized_batch = {
            key: value.to(device)
            for key, value in tokenized_batch.items()
        }

        with torch.inference_mode():
            generated_ids = model.generate(
                **tokenized_batch,
                num_beams=5,
                max_length=256,
            )

        decoded_texts = tokenizer.batch_decode(
            generated_ids,
            src=False,
        )

        processed_texts = processor.postprocess_batch(
            decoded_texts,
            lang=target_language,
        )

        for text, is_special in zip(
            processed_texts,
            batch_flags,
        ):
            if is_special:
                text = f"_*{text}*_."

            translated_texts.append(text)

    return translated_texts


#Integration
def process_document(
    image_path: Path,
    language: str = "en",
    pitch: float = -2,
    speed: float = 1.0,
) -> ReaderResult:
    """
    Run the complete document-reading pipeline.
    """

    page = detect_layout(
        image_path
    )

    texts = extract_text(
        page
    )

    captions = {}

    if page.regions.get("Picture"):
        captions = caption_images(
            page
        )

        texts = associate_images(
            page=page,
            texts=texts,
            captions=captions,
        )

    translated_texts = translate_text(
        texts=texts,
        language=language,
    )

    output_directory = (
        settings.create_output_directory()
    )

    audio_path = output_directory / (
        f"{image_path.stem}_{language}.wav"
    )

    text_to_audio(
        texts=translated_texts,
        output_path=audio_path,
        language=language,
        pitch=pitch,
        speed=speed,
    )

    page.text = translated_texts

    return ReaderResult(
        text=translated_texts,
        pages=[page],
        audio_path=audio_path,
    )
