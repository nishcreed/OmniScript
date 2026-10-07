from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import base64

@dataclass
class Region:
    label: str
    box: list[float]
    score: float

@dataclass
class PageResult:
    image_path: Path
    regions: dict[str, list[Region]]
    text: list[str] = field(default_factory=list)

@dataclass
class ReaderResult:
    text: list[str]
    pages: list[PageResult]
    audio_bytes: bytes

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "pages": [
                {
                    "regions": {
                        label: [
                            {
                                "box": region.box,
                                "score": region.score,
                            }
                            for region in regions
                        ]
                        for label, regions in page.regions.items()
                    },
                    "text": page.text,
                }
                for page in self.pages
            ],
            "audio": {
                "mime_type": "audio/wav",
                "encoding": "base64",
                "data": base64.b64encode(
                    self.audio_bytes
                ).decode("ascii"),
            },
        }