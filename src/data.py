from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    audio_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "pages": [
                {
                    "image_path": str(page.image_path),
                    "regions": {
                        label: [
                            {
                                "box": region.box,
                                "score": region.score
                            }
                            for region in regions
                        ]
                        for label, regions in page.regions.items()
                    },
                    "text": page.text
                }
                for page in self.pages
            ],
            "audio_path": (
                str(self.audio_path)
                if self.audio_path is not None
                else None
            )
        }