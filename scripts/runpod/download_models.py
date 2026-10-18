import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from src.config import settings

MODEL_REPOSITORIES = (
    "Salesforce/blip2-opt-2.7b",
    "Salesforce/blip2-itm-vit-g",
    "ai4bharat/indictrans2-en-indic-dist-200M",
)


def validate_json_files(repo_id: str, snapshot_path: str, files: list[str]) -> None:
    snapshot = Path(snapshot_path)

    for filename in files:
        if not filename.endswith(".json"):
            continue

        path = snapshot / filename
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            print(f"Refreshing invalid JSON: {repo_id}/{filename}", flush=True)
            repaired_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=snapshot.name,
                force_download=True,
            )
            # Fail startup if the refreshed file is still unreadable or invalid.
            json.loads(Path(repaired_path).read_text(encoding="utf-8"))

def download_layout() -> None:
    model_path = settings.layout_model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)

    downloaded_path = hf_hub_download(
        repo_id="DILHTWD/documentlayoutsegmentation_YOLOv8_ondoclaynet",
        filename=model_path.name,
        local_dir=str(model_path.parent),
    )

    print(f"Layout checkpoint ready: {downloaded_path}", flush=True)


def download_transformer_models() -> None:
    api = HfApi()

    for repo_id in MODEL_REPOSITORIES:
        print(f"Preparing {repo_id}", flush=True)

        files = api.list_repo_files(repo_id)

        has_safetensors = any(
            filename.endswith(".safetensors")
            for filename in files
        )
        weight_pattern = (
            "*.safetensors" if has_safetensors else "*.bin"
        )

        snapshot_path = snapshot_download(
            repo_id=repo_id,
            allow_patterns=[
                weight_pattern,
                "*.json",
                "*.model",
                "*.txt",
                "*.py"
            ],
            max_workers=4,
        )

        validate_json_files(repo_id, snapshot_path, files)
        print(f"Cached at: {snapshot_path}", flush=True)


def main() -> None:
    download_layout()
    download_transformer_models()


if __name__ == "__main__":
    main()


