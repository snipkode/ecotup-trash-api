# ecotup-trash-api

Flask REST API for Trash Classification using TFLite models (Sigmoid + Softmax).

## Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Classify trash image |

## Request

```
POST http://43.157.208.51:5001/predict
Content-Type: multipart/form-data
Field: image (jpg/png/bmp/webp)
```

## Response

```json
{
  "error": false,
  "category": "Anorganik",
  "confidence": 99.8,
  "sigmoid_scores": { "Organik": 0.2, "Anorganik": 99.8 },
  "sub_category": "Metal",
  "sub_confidence": 96.04,
  "softmax_scores": {
    "Glass": 0.41, "Metal": 96.04, "Paper": 0.43, "Plastic": 3.13
  }
}
```

## Classification Logic

```
Image → Sigmoid Model → Organik / Anorganik
                              │
                         Anorganik → Softmax Model → Glass / Metal / Paper / Plastic
```

## Models (not included in repo — too large for GitHub)

Place these files in the project root before building:
- `trashClassification_sigmoid.tflite` (88MB) — from ecotup_mobile
- `trashClassification_softmax.tflite` (88MB) — from ecotup_mobile

## Docker

```bash
# Build
docker build -t ecotup-trash-api:latest .

# Run
docker run -d \
  --name ecotup-trash-api \
  --restart unless-stopped \
  -p 5001:5001 \
  ecotup-trash-api:latest
```

## Dependencies

- Flask 3.0.0
- tflite-runtime 2.14.0
- Pillow 10.1.0
- numpy 1.26.0
