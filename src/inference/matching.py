"""Sentence-level image-text matching and caption placement."""

from PIL import Image
import torch

from src.data import PageResult
from src.model import load_itm_model


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
