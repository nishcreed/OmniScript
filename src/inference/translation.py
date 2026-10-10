"""English-to-Indic translation with narration marker preservation."""

from src.model import load_translation_model


def translate_text(
    texts: list[str],
    language: str,
    batch_size: int = 4,
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
        raise ValueError(f"Unsupported language: {language}")

    if batch_size < 1:
        raise ValueError("Translation batch size must be positive.")

    if not texts:
        return []

    import torch
    from IndicTransToolkit import IndicProcessor

    model, tokenizer = load_translation_model()
    processor = IndicProcessor(inference=True)

    target_language = language_codes[language]
    device = next(model.parameters()).device
    translated_texts: list[str] = []

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        special_flags = [
            text.startswith("_*") and text.endswith("*_.")
            for text in batch
        ]

        cleaned_batch = [
            text[2:-3] if special else text
            for text, special in zip(batch, special_flags)
        ]

        prepared_batch = processor.preprocess_batch(
            cleaned_batch,
            src_lang="eng_Latn",
            tgt_lang=target_language,
        )

        inputs = tokenizer(
            prepared_batch,
            padding="longest",
            truncation=False,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                num_beams=5,
                max_length=256,
            )

        decoded_texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translated_batch = processor.postprocess_batch(
            decoded_texts,
            lang=target_language,
        )

        for text, special in zip(translated_batch, special_flags):
            if special:
                text = f"_*{text}*_."

            translated_texts.append(text)

    return translated_texts
