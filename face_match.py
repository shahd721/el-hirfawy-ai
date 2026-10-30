
from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

DETECTOR_PATH = (
    BASE_DIR / "models" / "face_detection_yunet_2023mar.onnx"
)

RECOGNIZER_PATH = (
    BASE_DIR / "models" / "face_recognition_sface_2021dec.onnx"
)

ID_IMAGE_PATH = BASE_DIR / "test_images" / "fake_id.png"
SELFIE_IMAGE_PATH = BASE_DIR / "test_images" / "selfie.png"


# --------------------------------------------------
# 2. LOAD AI MODELS
# --------------------------------------------------

def load_models():
    detector = cv2.FaceDetectorYN.create(
        str(DETECTOR_PATH),
        "",
        (320, 320),
        0.8,
        0.3,
        5000
    )

    recognizer = cv2.FaceRecognizerSF.create(
        str(RECOGNIZER_PATH),
        ""
    )

    return detector, recognizer


# --------------------------------------------------
# 3. READ IMAGE AND DETECT FACE
# --------------------------------------------------

def detect_face(image_path, detector):
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height, width = image.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(image)

    if faces is None or len(faces) == 0:
        raise ValueError(
            f"No face detected in {image_path.name}"
        )

    if len(faces) > 1:
        raise ValueError(
            f"Multiple faces detected in {image_path.name}. "
            "Use an image with one clear face."
        )

    face = faces[0]

    return image, face


# --------------------------------------------------
# 4. COMPARE TWO FACES
# --------------------------------------------------

def compare_faces(id_image_path, selfie_image_path):
    detector, recognizer = load_models()

    id_image, id_face = detect_face(
        id_image_path,
        detector
    )

    selfie_image, selfie_face = detect_face(
        selfie_image_path,
        detector
    )

    id_aligned = recognizer.alignCrop(
        id_image,
        id_face
    )

    selfie_aligned = recognizer.alignCrop(
        selfie_image,
        selfie_face
    )

    id_features = recognizer.feature(id_aligned)
    selfie_features = recognizer.feature(selfie_aligned)

    score = recognizer.match(
        id_features,
        selfie_features,
        cv2.FaceRecognizerSF_FR_COSINE
    )

    return float(score)


# --------------------------------------------------
# 5. RUN TEST
# --------------------------------------------------


DIFFERENT_PERSON_PATH = BASE_DIR / "test_images" / "different_person.png"

if __name__ == "__main__":
    tests = [
        ("Card vs matching selfie", SELFIE_IMAGE_PATH),
        ("Card vs different person", DIFFERENT_PERSON_PATH),
    ]

    for label, selfie_path in tests:
        print(f"\n--- {label} ---")

        try:
            score = compare_faces(ID_IMAGE_PATH, selfie_path)
            print(f"Cosine similarity score: {score:.4f}")

        except (ValueError, cv2.error) as error:
            print(f"Test failed: {error}")

