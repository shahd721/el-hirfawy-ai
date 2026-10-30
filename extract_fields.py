
import json
import re
from pathlib import Path

# File paths
project_dir = Path(__file__).parent
input_path = project_dir / "ocr_result.json"
output_path = project_dir / "extracted_fields.json"


def normalize_digits(text):
    """Convert Arabic digits to English digits."""
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"

    for index, digit in enumerate(arabic_digits):
        text = text.replace(digit, str(index))

    for index, digit in enumerate(persian_digits):
        text = text.replace(digit, str(index))

    return text


def find_national_id(text_blocks):
    """Find a possible 14-digit number in detected text."""
    for block in text_blocks:
        text = normalize_digits(block["text"])
        digits_only = re.sub(r"\D", "", text)

        if len(digits_only) == 14:
            return digits_only

    return None


def main():
    # 1. Check input file
    if not input_path.exists():
        print("ERROR: ocr_result.json was not found.")
        return

    # 2. Read OCR output locally
    with input_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    text_blocks = data.get("text_blocks", [])

    # 3. Keep the OCR text for manual review
    all_text = [
        block["text"]
        for block in text_blocks
        if block.get("text", "").strip()
    ]

    # 4. Find a possible national ID
    national_id = find_national_id(text_blocks)

    # 5. Create a structured result
    extracted = {
        "document_type": "egyptian_national_id",
        "fields": {
            "name": None,
            "address": None,
            "national_id": national_id
        },
        "ocr_text_for_review": all_text,
        "field_extraction_status": {
            "name": "needs_field_detection",
            "address": "needs_field_detection",
            "national_id": (
                "candidate_found"
                if national_id
                else "not_detected"
            )
        },
        "review_required": True
    }

    # 6. Save locally
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            extracted,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("Field extraction stage completed.")
    print("Name: not determined")
    print("Address: not determined")
    print(
        "National ID:",
        "candidate found" if national_id else "not detected"
    )
    print("Output saved to:", output_path.name)


if __name__ == "__main__":
    main()
