import os
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    layout_model_path: Path = Path(
        os.getenv(
            "OMNISCRIPT_LAYOUT_MODEL",
            "models/yolov8x-doclaynet-epoch64-imgsz640-initiallr1e-4-finallr1e-5.pt"
        )
    )

    output_directory: Path = Path(
        os.getenv(
            "OMNISCRIPT_OUTPUT_DIR",
            "outputs"
        )
    )

    device: str = os.getenv(
        "OMNISCRIPT_DEVICE",
        "auto"
    )

    tesseract_command: str = os.getenv(
        "TESSERACT_CMD",
        "tesseract"
    )

    def create_output_directory(self) -> Path:
        self.output_directory.mkdir(
            parents=True,
            exist_ok=True
        )
        return self.output_directory

settings = Settings()