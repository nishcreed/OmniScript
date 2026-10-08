from pathlib import Path
from tempfile import TemporaryDirectory

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

import asyncio
import os
from typing import Any, Literal

if os.getenv("OMNISCRIPT_MOCK", "0") == "1":
    from src.mock_inference import process_document
else:
    from src.inference import process_document

api = FastAPI(title="OmniScript API")

processing_lock = asyncio.Lock() #one API request at a time

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg"}

Language = Literal["en", "ta", "hi", "te", "ml", "kn"]

@api.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }

def process_upload(
    image_bytes: bytes,
    suffix: str,
    language: str,
    pitch: float,
    speed: float
) -> dict[str, Any]:
    # Run blocking processing and clean up the uploaded image.

    with TemporaryDirectory(prefix="omniscript-") as directory:
        image_path = Path(directory) / f"upload{suffix}"
        image_path.write_bytes(image_bytes)

        result = process_document(
            image_path=image_path,
            language=language,
            pitch=pitch,
            speed=speed
        )

        return result.as_dict()



@api.post("/process")
async def process(
    image: UploadFile = File(...),
    language: Language = Form("en"),
    pitch: float = Form(-2, ge=-12, le=12),
    speed: float = Form(1.0, ge=0.5, le=2.0)
):
    """
    Process one uploaded document image.
    """

    try:
        suffix = Path(image.filename or "").suffix.lower()

        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail="Upload a PNG or JPEG image."
            )

        image_bytes = await image.read(MAX_UPLOAD_BYTES + 1)

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file is empty."
            )

        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="The image must be at most 20 MB."
            )
        
    finally:
        await image.close()

    try:
        async with processing_lock:
            return await run_in_threadpool(
                process_upload,
                image_bytes,
                suffix,
                language,
                pitch,
                speed
            )
    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        ) from error
