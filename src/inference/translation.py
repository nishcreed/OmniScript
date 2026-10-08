"""English-to-Indic translation with narration marker preservation."""

from src.model import load_translation_model


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
