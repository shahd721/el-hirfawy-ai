
# El-Hirfawy AI Service

Python AI service for OCR and face similarity comparison.

## Requirements

- Python 3.13
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Run the API

```bash
uvicorn main:app --reload
```

## API Documentation

Open:

http://127.0.0.1:8000/docs

## Face Matching

Endpoint: POST /face-match

Content-Type: multipart/form-data

Fields:
- id_image: ID card image
- selfie_image: selfie image

The response returns a similarity score.
The score is not a probability and does not by itself
confirm identity or detect spoofing.

## OCR

Use the OCR endpoint documented in /docs.
