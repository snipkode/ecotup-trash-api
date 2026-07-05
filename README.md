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

## Memory Usage

Diukur langsung di server produksi menggunakan `docker stats`.

**Pemakaian RAM per container:**

| Container | RAM | % dari Total |
|-----------|-----|-------------|
| `ecotup-trash-api` | **245 MB** | 12.5% |
| `mysql8` | 128 MB | 6.5% |
| `phpmyadmin` | 52 MB | 2.6% |
| `ecotup-finding-driver` | 3.6 MB | 0.2% |
| **Total semua service** | **~430 MB** | **~22%** |

**Breakdown memory ecotup-trash-api (245 MB):**

```
Sigmoid model  (trashClassification_sigmoid.tflite)  : ~88 MB
Softmax model  (trashClassification_softmax.tflite)  : ~88 MB
Python runtime + Flask + numpy + Pillow              : ~65 MB
XNNPACK delegate (CPU acceleration)                  : ~4 MB
──────────────────────────────────────────────────────────────
Total idle                                           : ~245 MB
```

**Memory saat inference:**

| Kondisi | RAM |
|---------|-----|
| Idle (model loaded, tidak ada request) | 245 MB |
| Saat 1 request aktif | 245 MB + ~0.75 MB |
| Saat 3 concurrent requests | ~248 MB |
| Spike per inference | **+0.75 MB** (temp buffer) |

> Memory inference sangat efisien — hanya naik ~0.75 MB per request karena buffer preprocessing gambar (255×255×3×4 bytes = ~780 KB) dibuat dan langsung dibuang setelah inference selesai.

**Kondisi server saat ini:**

```
Total RAM     : 1.9 GB
Terpakai      : 1.1 GB  (termasuk OS + semua service)
Available     : ~790 MB
Swap terpakai : 1.0 GB / 1.9 GB
```

> ⚠️ Server sudah menggunakan swap 1GB. Jika traffic meningkat atau ada service tambahan, pertimbangkan upgrade RAM ke minimal 4GB agar tidak swap-heavy.

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
