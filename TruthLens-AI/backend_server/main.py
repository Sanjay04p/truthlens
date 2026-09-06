import shutil
import tempfile
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# 1. Import both pipelines
from pipelines.text_pipeline import TextVerificationPipeline
from pipelines.image_pipeline import ImageVerificationPipeline

app = FastAPI(
    title="TruthLens AI Engine",
    description="Multimodal Fake News Detection API",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Global Engine Initialization
print("[*] Booting up AI Engines. Please wait...")
text_engine = TextVerificationPipeline()
image_engine = ImageVerificationPipeline()
print("[*] Both Text and Image engines initialized successfully.")

# # 3. Pydantic Models
# class ClaimPayload(BaseModel):
#     text: str

# 4. Text Verification Endpoint
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from fastapi import HTTPException


# ============================================================
# Structured Schemas
# ============================================================

class ClaimPayload(BaseModel):
    text: str = Field(
        ...,
        min_length=3,
        description="Text claim to verify"
    )


class SourceInfo(BaseModel):
    name: str
    domain: str = ""
    url: str = ""
    role: str = ""


class VerificationResponse(BaseModel):
    tier: int
    label: str
    confidence_score: str
    breakdown: Dict[str, Any]
    reason: str
    sources: List[SourceInfo] = []
    verification_type: str = ""


# ============================================================
# Text Verification Endpoint
# ============================================================

@app.post(
    "/verify-text",
    response_model=VerificationResponse
)
def verify_text_endpoint(payload: ClaimPayload):

    try:
        # --------------------------------------------------------
        # Clean and validate input
        # --------------------------------------------------------

        clean_text = payload.text.strip()

        if not clean_text:
            raise HTTPException(
                status_code=400,
                detail="Text claim cannot be empty."
            )

        # --------------------------------------------------------
        # Run TruthLens verification pipeline
        # --------------------------------------------------------

        result = text_engine.verify_claim(clean_text)

        # --------------------------------------------------------
        # Handle internal pipeline errors
        # --------------------------------------------------------

        if result.get("label") == "Error":
            raise HTTPException(
                status_code=502,
                detail=result.get(
                    "reason",
                    "Downstream inference provider failed."
                )
            )

        # --------------------------------------------------------
        # Return structured verification result
        # --------------------------------------------------------

        return result

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Text verification failed: {str(e)}"
        )

# 5. Image Verification Endpoint
@app.post("/verify-image")
def verify_image_endpoint(file: UploadFile = File(...)):
    """
    Accepts an uploaded image file (JPEG, PNG, WebP), saves it to a temporary
    file, and executes the dual-strategy (HF + Sightengine) image pipeline.
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/jpg"]
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file.content_type}. Use JPEG, PNG, or WebP."
        )

    # Determine file extension safely
    suffix = Path(file.filename).suffix if file.filename else ".jpg"
    if not suffix:
        suffix = ".jpg"

    temp_path = None
    try:
        # Write incoming bytes to a temporary file on disk for the pipeline
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = Path(temp_file.name)

        # Run verification through the dual-strategy vision pipeline
        result = image_engine.verify_image(str(temp_path))
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

    finally:
        # Ensure cleanup of the temporary file to prevent disk exhaustion
        if temp_path and temp_path.exists():
            temp_path.unlink()

# 6. Server Execution
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)