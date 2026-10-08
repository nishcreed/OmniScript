"""OCR preprocessing and extraction of text and table descriptions."""

from PIL import Image

from src.config import settings
from src.data import PageResult, Region
from .layout import sort_regions


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
