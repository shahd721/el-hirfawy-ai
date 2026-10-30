
import os
import re
import secrets
from pathlib import Path

import cv2
import easyocr
import numpy as np

from dotenv import load_dotenv
import fastapi

from face_match import compare_faces


# =========================================================
# 1. CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

# Load API key from the .env file beside main.py
load_dotenv(BASE_DIR / ".env")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

app = fastapi.FastAPI(
    title="El-Hirfawy AI Service",
    description="OCR and Face Matching API",
    version="1.0.0",
)


# =========================================================
# 2. OCR READER
# =========================================================

# Initialize once when the application starts.
# The first initialization may download OCR model files.
ocr_reader = easyocr.Reader(["ar", "en"], gpu=False)


# =========================================================
# 3. API KEY AUTHENTICATION
# =========================================================

def verify_api_key(
    x_api_key: str = fastapi.Header(..., alias="X-Api-Key")
):
    expected_key = os.getenv("AI_API_KEY")

    if not expected_key:
        raise fastapi.HTTPException(
            status_code=500,
            detail="Server API key is missing. Check the .env file.",
        )

    if not secrets.compare_digest(x_api_key, expected_key):
        raise fastapi.HTTPException(
            status_code=401,
            detail="Invalid API key.",
        )

    return True


# =========================================================
# 4. IMAGE VALIDATION
# =========================================================

async def read_image_bytes(image: fastapi.UploadFile) -> bytes:
    extension = Path(image.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise fastapi.HTTPException(
            status_code=400,
            detail="Upload a JPG, JPEG, PNG, or WEBP image.",
        )

    image_bytes = await image.read(MAX_FILE_SIZE + 1)

    if not image_bytes:
        raise fastapi.HTTPException(
            status_code=400,
            detail="The uploaded image is empty.",
        )

    if len(image_bytes) > MAX_FILE_SIZE:
        raise fastapi.HTTPException(
            status_code=413,
            detail="Image size must not exceed 5 MB.",
        )

    return image_bytes


def decode_image(image_bytes: bytes):
    array = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)

    if image is None:
        raise fastapi.HTTPException(
            status_code=422,
            detail="The uploaded file is not a valid image.",
        )

    return image


# =========================================================
# 5. NATIONAL ID CANDIDATE EXTRACTION
# =========================================================

def normalize_digits(text: str) -> str:
    """Convert Arabic-Indic digits to English digits."""

    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"

    translation = {}

    for index, digit in enumerate(arabic_digits):
        translation[ord(digit)] = str(index)

    for index, digit in enumerate(persian_digits):
        translation[ord(digit)] = str(index)

    return text.translate(translation)


def find_national_id_candidate(text_blocks: list[str]):
    """
    Find a possible 14-digit Egyptian national ID in OCR output.

    This only identifies a candidate; it does not verify the ID.
    """

    normalized_text = normalize_digits(" ".join(text_blocks))

    # Try to find a continuous sequence of 14 digits.
    candidates = re.findall(
        r"(?<!\d)\d{14}(?!\d)",
        normalized_text,
    )

    if candidates:
        return candidates[0]

    # Fallback: combine digit groups only when they are separated
    # by non-digit characters, then search for a 14-digit sequence.
    digit_groups = re.findall(r"\d+", normalized_text)
    combined_digits = "".join(digit_groups)

    if len(combined_digits) == 14:
        return combined_digits

    return None


# =========================================================
# 6. HOME ENDPOINT
# =========================================================

@app.get("/")
def home():
    return {
        "success": True,
        "service": "El-Hirfawy AI",
        "message": "AI service is running.",
        "endpoints": {
            "health": "/health",
            "ocr": "/ocr",
            "face_match": "/face-match",
            "documentation": "/docs",
        },
    }


# =========================================================
# 7. HEALTH CHECK
# =========================================================

@app.get("/health")
def health():
    return {
        "success": True,
        "status": "running",
        "api_key_configured": bool(os.getenv("AI_API_KEY")),
    }


# =========================================================
# 8. OCR ENDPOINT
# =========================================================

@app.post("/ocr")
async def ocr_endpoint(
    id_image: fastapi.UploadFile = fastapi.File(...),
    _authenticated: bool = fastapi.Depends(verify_api_key),
):
    image_bytes = await read_image_bytes(id_image)
    image = decode_image(image_bytes)

    try:
        results = ocr_reader.readtext(image)

        text_blocks = [
            result[1]
            for result in results
            if len(result) >= 2
        ]

        extracted_text = "\n".join(text_blocks)

        national_id_candidate = find_national_id_candidate(
            text_blocks
        )

        return {
            "success": True,
            "extracted_text": extracted_text,
            "national_id_candidate": national_id_candidate,
            "review_required": True,
            "message": (
                "OCR completed. Verify the extracted information "
                "against the original document."
            ),
        }

    except fastapi.HTTPException:
        raise

    except Exception:
        raise fastapi.HTTPException(
            status_code=500,
            detail="OCR processing failed.",
        )


# =========================================================
# 9. FACE MATCHING ENDPOINT
# =========================================================

@app.post("/face-match")
async def face_match_endpoint(
    id_image: fastapi.UploadFile = fastapi.File(...),
    selfie_image: fastapi.UploadFile = fastapi.File(...),
    _authenticated: bool = fastapi.Depends(verify_api_key),
):
    id_bytes = await read_image_bytes(id_image)
    selfie_bytes = await read_image_bytes(selfie_image)

    # Validate that both files are readable images.
    decode_image(id_bytes)
    decode_image(selfie_bytes)

    id_ext = Path(id_image.filename or "").suffix.lower()
    selfie_ext = Path(selfie_image.filename or "").suffix.lower()

    try:
        # Temporary files are deleted automatically afterwards.
        import tempfile

        with tempfile.TemporaryDirectory() as temp_dir:
            id_path = Path(temp_dir) / f"id{ id_ext }"
            selfie_path = Path(temp_dir) / f"selfie{ selfie_ext }"

            id_path.write_bytes(id_bytes)
            selfie_path.write_bytes(selfie_bytes)

            score = compare_faces(id_path, selfie_path)

        return {
            "success": True,
            "similarity_score": round(float(score), 4),
            "decision": "not_calibrated",
            "message": (
                "Similarity calculated. A validated threshold "
                "is required before making an identity decision."
            ),
        }

    except fastapi.HTTPException:
        raise

    except ValueError as error:
        raise fastapi.HTTPException(
            status_code=422,
            detail=str(error),
        )

    except cv2.error:
        raise fastapi.HTTPException(
            status_code=422,
            detail="Could not process one or both images.",
        )

    except Exception:
        raise fastapi.HTTPException(
            status_code=500,
            detail="Face matching failed. Check the model files and server logs.",
        )





 
  