
import json
import re
from pathlib import Path
from tkinter import Tk, filedialog

import easyocr


def normalize_digits(text):
    """Convert Arabic and Persian digits to English digits."""
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    persian_digits = "۰۱۲۳۴۵۶۷۸۹"

    for index, digit in enumerate(arabic_digits):
        text = text.replace(digit, str(index))

    for index, digit in enumerate(persian_digits):
        text = text.replace(digit, str(index))

    return text


def extract_national_id(text):
    """Find a possible 14-digit national ID."""
    normalized = normalize_digits(text)

    # Allow spaces or hyphens between digits.
    pattern = r"(?<!\d)\d(?:[\s-]*\d){13}(?!\d)"
    matches = re.findall(pattern, normalized)

    for match in matches:
        candidate = re.sub(r"[\s-]", "", match)

        if len(candidate) == 14:
            return candidate

    return None


def main():
    # 1. Select an image
    root = Tk()
    root.withdraw()

    image_path = filedialog.askopenfilename(
        title="Choose an ID image",
        filetypes=[
            ("Image files", "*.png *.jpg *.jpeg *.bmp")
        ]
    )

    root.destroy()

    if not image_path:
        print("No image selected.")
        return

    # 2. Load OCR model
    print("Loading Arabic and English OCR...")
    reader = easyocr.Reader(["ar", "en"], gpu=False)

    # 3. Read text
    print("Reading image...")
    results = reader.readtext(
        image_path,
        detail=1,
        paragraph=False
    )

    # 4. Organize detected text blocks
    blocks = []

    for result in results:
        box, detected_text, confidence = result

        # Get approximate top-left position
        x = min(point[0] for point in box)
        y = min(point[1] for point in box)

        if detected_text.strip():
            blocks.append({
                "text": detected_text.strip(),
                "confidence": round(float(confidence), 3),
                "x": round(float(x), 1),
                "y": round(float(y), 1)
            })

    # Sort approximately from top to bottom, then left to right
    blocks.sort(key=lambda item: (item["y"], item["x"]))

    # 5. Combine detected text
    full_text = "\n".join(
        block["text"] for block in blocks
    )

    # 6. Extract a possible national ID
    national_id = extract_national_id(full_text)

    # 7. Build structured result
    output = {
        "document_type": "egyptian_national_id",
        "source_image": Path(image_path).name,
        "full_text": full_text,
        "national_id_candidate": national_id,
        "text_blocks": blocks,
        "review_required": True,
        "notes": [
            "OCR output must be checked against the source image.",
            "Name and address extraction are not implemented yet.",
            "A detected number is not proof of identity or validity."
        ]
    }

    # 8. Save JSON next to this Python file
    output_path = Path(__file__).with_name("ocr_result.json")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            output,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 9. Show a summary
    print("\n========== STRUCTURED OCR RESULT ==========")
    print("Detected text blocks:", len(blocks))
    print("Possible national ID:", national_id)
    print("JSON saved to:", output_path)

    print("\nDetected text:")
    print(full_text)

    print("\nOCR stage completed.")


if __name__ == "__main__":
    main()
