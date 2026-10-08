"""Document layout detection and region ordering."""

from collections.abc import Iterable
from pathlib import Path

from src.data import PageResult, Region
from src.model import load_layout_model


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
