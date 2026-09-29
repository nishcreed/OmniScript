from collections.abc import Iterable

from src.data import Region

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