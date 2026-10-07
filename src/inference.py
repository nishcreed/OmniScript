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
    """Resolve overlaps using the notebook's class preferences."""

    kept_regions: list[Region] = []

    def should_replace(existing: Region, incoming: Region) -> bool:
        # Prefer captions over other region types.
        if existing.label == "Caption":
            return False

        if incoming.label == "Caption":
            return True

        # Prefer page headers when they overlap page footers.
        if (
            existing.label == "Page-footer"
            and incoming.label == "Page-header"
        ):
            return True

        if (
            existing.label == "Page-header"
            and incoming.label == "Page-footer"
        ):
            return False

        # Otherwise, prefer the higher confidence score.
        return incoming.score > existing.score

    for region in regions:
        for index, existing in enumerate(kept_regions):
            overlap = calculate_iou(existing.box, region.box)

            if overlap > iou_threshold:
                if should_replace(existing, region):
                    kept_regions[index] = region

                break
        else:
            kept_regions.append(region)

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
    """Read text and table descriptions in page order."""

    import pytesseract

    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_command

    content_labels = {
        "Text",
        "Title",
        "Section-header",
        "List-item",
        "Footnote",
        "Formula",
        "Table",
    }

    headers = sort_regions(page.regions.get("Page-header", []))

    content = sort_regions(
        region
        for label, regions in page.regions.items()
        if label in content_labels
        for region in regions
    )

    footers = sort_regions(page.regions.get("Page-footer", []))

    extracted_text: list[str] = []

    with Image.open(page.image_path) as source:
        image = source.convert("RGB")
        ocr_cache: dict[tuple[float, ...], str] = {}

        def read_region(region: Region) -> str:
            key = tuple(region.box)

            if key in ocr_cache:
                return ocr_cache[key]

            x1, y1, x2, y2 = region.box

            box = (
                max(0, min(image.width, x1)),
                max(0, min(image.height, y1)),
                max(0, min(image.width, x2)),
                max(0, min(image.height, y2)),
            )

            if box[2] <= box[0] or box[3] <= box[1]:
                ocr_cache[key] = ""
                return ""

            crop = image.crop(box)
            text = clean_text(
                pytesseract.image_to_string(preprocess_image(crop))
            )

            ocr_cache[key] = text
            return text

        def read_table_description(table: Region) -> str:
            tx1, ty1, tx2, ty2 = table.box

            # Prefer Caption regions; fall back to nearby Text.
            for label in ("Caption", "Text"):
                candidates: list[tuple[float, Region]] = []

                for candidate in page.regions.get(label, []):
                    cx1, cy1, cx2, cy2 = candidate.box

                    horizontal_overlap = (
                        min(tx2, cx2) > max(tx1, cx1)
                    )

                    # Candidate must be above or below the table.
                    if cy2 <= ty1:
                        distance = ty1 - cy2
                    elif cy1 >= ty2:
                        distance = cy1 - ty2
                    else:
                        continue

                    if horizontal_overlap and distance <= 40:
                        candidates.append((distance, candidate))

                candidates.sort(key=lambda item: item[0])

                for _, candidate in candidates:
                    text = read_region(candidate)

                    if text:
                        return f"_*A table: {text}*_."

            return "_*A table*_."

        for region in headers + content + footers:
            if region.label == "Table":
                extracted_text.append(read_table_description(region))
                continue

            text = read_region(region)

            if not text:
                continue

            if region.label != "Text":
                text = f"_*{region.label}: {text}*_."

            extracted_text.append(text)

    page.text = extracted_text
    return extracted_text


#Caption Generation
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


#Image-Text Matching
def associate_images(
    page: PageResult,
    texts: list[str],
    captions: dict[int, str],
) -> list[str]:
    """Place picture captions beside the best matching sentence."""

    from nltk.tokenize import sent_tokenize

    sentences: list[str] = []

    for text in texts:
        text = text.strip()

        if not text:
            continue

        if text.startswith("_*") and text.endswith("*_."):
            sentences.append(text)
        else:
            sentences.extend(sent_tokenize(text))

    # Store captions separately to preserve their order when several
    # pictures match the same sentence.
    first_captions: list[str] = []
    last_captions: list[str] = []
    captions_after: dict[int, list[str]] = {}

    candidates = [
        (index, sentence)
        for index, sentence in enumerate(sentences)
        if not (
            sentence.startswith("_*")
            and sentence.endswith("*_.")
        )
    ]

    picture_regions = page.regions.get("Picture", [])

    if not picture_regions:
        return sentences

    with Image.open(page.image_path) as source:
        image = source.convert("RGB")

        for picture_index, region in enumerate(
            picture_regions,
            start=1,
        ):
            caption = captions.get(picture_index)

            if not caption:
                continue

            if region.box[1] < 50:
                first_captions.append(caption)
                continue

            if region.box[3] > image.height - 50:
                last_captions.append(caption)
                continue

            if not candidates:
                last_captions.append(caption)
                continue

            # The loader caches the model after the first call.
            model, processor = load_itm_model()
            cropped_image = image.crop(tuple(region.box))

            best_sentence_index = candidates[0][0]
            best_score = float("-inf")

            for sentence_index, sentence in candidates:
                inputs = processor(
                    images=cropped_image,
                    text=sentence,
                    return_tensors="pt",
                )

                inputs = {
                    key: (
                        value.to(
                            device=model.device,
                            dtype=model.dtype,
                        )
                        if value.is_floating_point()
                        else value.to(model.device)
                    )
                    for key, value in inputs.items()
                }

                with torch.inference_mode():
                    result = model(
                        **inputs,
                        use_image_text_matching_head=True,
                    )

                score = torch.softmax(
                    result.logits_per_image,
                    dim=-1,
                )[0, 1].item()

                if score > best_score:
                    best_score = score
                    best_sentence_index = sentence_index

            captions_after.setdefault(
                best_sentence_index,
                [],
            ).append(caption)

    output_texts = list(first_captions)

    for sentence_index, sentence in enumerate(sentences):
        output_texts.append(sentence)
        output_texts.extend(
            captions_after.get(sentence_index, [])
        )

    output_texts.extend(last_captions)

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