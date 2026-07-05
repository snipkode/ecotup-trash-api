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

## Performance (Server Benchmark)

Diukur langsung di server produksi dengan 10 request berturut-turut.

**Spesifikasi Server:**
| Komponen | Detail |
|----------|--------|
| CPU | Intel Xeon Gold 6231C @ 3.20GHz (2 vCPU) |
| RAM | 1.9 GB total / ~800 MB available |
| Inference | CPU only (no GPU) |

**Hasil Benchmark — `POST /predict` (10 requests):**

| Request | Waktu |
|---------|-------|
| #1 (cold start) | 245 ms |
| #2 | 199 ms |
| #3 | 201 ms |
| #4 | 211 ms |
| #5 | 202 ms |
| #6 | 206 ms |
| #7 | 201 ms |
| #8 | 198 ms |
| #9 | 202 ms |
| #10 | 196 ms |

**Ringkasan:**
| Metrik | Nilai |
|--------|-------|
| Cold start (pertama) | ~245 ms |
| Rata-rata (steady) | ~201 ms |
| Minimum | ~196 ms |
| Maximum (steady) | ~211 ms |

> **Catatan:** Request pertama sedikit lebih lambat karena JIT warm-up XNNPACK delegate. Request berikutnya stabil di ~200ms.

**Breakdown waktu per tahap (estimasi):**
```
Upload gambar ke server    :  ~5–20 ms  (tergantung ukuran & jaringan)
Preprocessing (resize+norm):  ~10–20 ms
Sigmoid inference          :  ~80–100 ms
Softmax inference (jika    :  ~80–100 ms  (hanya jika Anorganik)
  Anorganik)
JSON serialization         :  <1 ms
────────────────────────────────────────
Total Organik              :  ~150–170 ms
Total Anorganik            :  ~200–220 ms
```

**Perbandingan dengan on-device (Android):**
| | On-device (TFLite Android) | Flask API (server) |
|-|---------------------------|-------------------|
| Latency | ~50–150 ms | ~200 ms + network |
| Internet | Tidak perlu | Wajib |
| RAM HP | ~200 MB | 0 |
| RAM Server | 0 | ~400 MB |
| Cocok untuk | Real-time scan | Integrasi backend/web |

## Dependencies

- Flask 3.0.0
- tflite-runtime 2.14.0
- Pillow 10.1.0
- numpy 1.26.0
