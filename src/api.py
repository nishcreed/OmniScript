from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from src.inference import process_document


api = FastAPI(title="OmniScript API")

@api.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }

@api.post("/process")
async def process(
    image: UploadFile = File(...),
    language: str = Form("en"),
    pitch: float = Form(-2),
    speed: float = Form(1.0)
):
    """
    Process one uploaded document image.
    """

    filename = image.filename or "upload.png"
    suffix = Path(filename).suffix or ".png"

    image_bytes = await image.read()

    try:
        with NamedTemporaryFile(
            suffix=suffix
        ) as temporary_file:
            temporary_file.write(image_bytes)

            temporary_file.flush()

            result = process_document(
                image_path=Path(
                    temporary_file.name
                ),
                language=language,
                pitch=pitch,
                speed=speed
            )

            return result.as_dict()
        
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error