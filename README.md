# ForexAI — USD/IDR Prediction Engine

Platform machine learning komparatif untuk prediksi nilai tukar USD/IDR menggunakan 5 algoritma berbeda.

## Algoritma

| Model | Tipe | Keterangan |
|-------|------|-----------|
| Linear Regression | Statistik | Baseline klasik |
| ANN | Deep Learning | 3 hidden layer + dropout |
| LSTM | Recurrent NN | Time-series specialist |
| K-Means | Clustering | Identifikasi rezim pasar |
| Backpropagation | Custom NN | Implementasi from scratch |

## Fitur

- ✅ Prediksi tanggal historis (2001–2025) DAN masa depan (2026+)
- ✅ 5 model berjalan paralel
- ✅ Identifikasi rezim pasar (K-Means)
- ✅ REST API endpoint
- ✅ Health check untuk Railway

## Struktur Proyek

```
usd_idr_app/
├── app/
│   ├── app.py              # Flask application
│   └── templates/          # HTML templates
│       ├── base.html
│       ├── index.html
│       ├── predict.html
│       ├── compare.html
│       └── about.html
├── data/
│   └── usd_idr_daily.csv   # Dataset historis
├── models/
│   ├── scaler.pkl
│   ├── linear_regression.pkl
│   ├── ann_model.h5
│   ├── lstm_model.h5
│   ├── kmeans_model.pkl
│   ├── backpropagation_model.pkl
│   └── all_metrics.pkl
├── Procfile                # Gunicorn command
├── railway.toml            # Railway config
├── requirements.txt
└── run.py
```

## Menjalankan Lokal

```bash
pip install -r requirements.txt
cd app
python app.py
# Buka http://localhost:5001
```

## Deploy ke Railway

1. Push ke GitHub
2. Connect repo di [railway.app](https://railway.app)
3. Railway otomatis deteksi `railway.toml` dan deploy
4. Set environment variables jika diperlukan (PORT otomatis di-set Railway)

## API Endpoints

- `POST /api/predict` — Prediksi berdasarkan tanggal: `{"year":2025,"month":12,"day":31}`
- `GET /api/status` — Status semua model
- `GET /api/latest_rate` — Kurs terbaru dari dataset
- `GET /health` — Health check

## Catatan Penting

- Dataset: data harian USD/IDR dari Yahoo Finance (2001–Oktober 2025)
- Untuk tanggal masa depan: model menggunakan 60 hari terakhir dari dataset sebagai basis fitur historis
- `tensorflow-cpu` digunakan di requirements.txt untuk deployment yang lebih ringan dan stabil di Railway
