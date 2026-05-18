# 🎓 Prediksi Kelulusan Mahasiswa — Proyek UTS Kecerdasan Buatan

Sistem prediksi kelulusan mahasiswa berbasis **Machine Learning** menggunakan 5 algoritma:
**Logistic Regression, ANN (MLP), RNN/LSTM, K-Means Clustering, dan Backpropagation**.

---

## 📁 Struktur Project

```
student_prediction/
├── data/
│   └── student_performance_prediction.csv   # Dataset (40.000 data)
├── notebooks/
│   └── train_models.py                      # Training semua model
├── models/
│   ├── scaler.pkl                           # StandardScaler tersimpan
│   ├── logistic_regression.pkl             # Model Logistic Regression
│   ├── ann_model.pkl                        # Model ANN (MLP)
│   ├── rnn_model.pkl                        # Model RNN/LSTM (Deep MLP)
│   ├── kmeans_model.pkl                     # Model K-Means Clustering
│   ├── backpropagation.pkl                  # Model Backpropagation
│   ├── pca.pkl                              # PCA untuk visualisasi
│   ├── results.json                         # Hasil evaluasi semua model
│   ├── meta.json                            # Metadata model terbaik
│   └── prep_meta.json                       # Metadata preprocessing
├── app/
│   ├── app.py                               # Flask web application
│   ├── static/
│   │   ├── css/style.css                    # Custom CSS
│   │   ├── js/main.js                       # JavaScript
│   │   └── img/                             # Grafik EDA & evaluasi
│   └── templates/
│       ├── base.html                        # Base template (navbar, footer)
│       ├── index.html                       # Halaman utama (form input)
│       ├── result.html                      # Halaman hasil prediksi
│       ├── compare.html                     # Perbandingan model
│       ├── eda.html                         # EDA & visualisasi
│       └── about.html                       # Tentang proyek
├── docs/
│   ├── laporan_jurnal.md                    # Laporan jurnal ilmiah
│   └── panduan_presentasi.md               # Panduan & script presentasi
├── requirements.txt                         # Dependensi Python
├── Procfile                                 # Deploy ke Railway/Render
└── README.md                                # Dokumentasi ini
```

---

## ⚙️ Cara Menjalankan Project

### 1. Clone / Download Project
```bash
git clone <repo-url>
cd student_prediction
```

### 2. Install Dependensi
```bash
pip install -r requirements.txt
```

### 3. Training Model (wajib dijalankan pertama)
```bash
python notebooks/train_models.py
```
Script ini akan:
- Membaca dataset dari `data/`
- Preprocessing & EDA
- Training 5 model ML
- Menyimpan model ke `models/`
- Menyimpan grafik ke `app/static/img/`

### 4. Jalankan Flask App
```bash
cd app
python app.py
```
Buka browser: **http://localhost:5000**

---

## 🤖 Algoritma Machine Learning

| No | Algoritma | Akurasi | Keterangan |
|----|-----------|---------|------------|
| 1 | Logistic Regression | 99.88% | Sederhana, interpretable |
| 2 | ANN (MLP) | 99.75% | 2 hidden layer, Adam optimizer |
| 3 | RNN/LSTM (Deep MLP) | 99.61% | 5 hidden layer, tanh, sequential |
| 4 | K-Means Clustering | 50.85% | Unsupervised, Silhouette=0.22 |
| 5 | **Backpropagation** | **99.91%** | **Model terbaik**, SGD optimizer |

---

## 📊 Metrik Evaluasi

- **Accuracy**: Persentase prediksi benar dari total data
- **MAE** (Mean Absolute Error): Rata-rata kesalahan prediksi
- **RMSE** (Root Mean Square Error): Akar rata-rata kuadrat kesalahan
- **R² Score**: Seberapa baik model menjelaskan variansi data
- **Confusion Matrix**: TP, TN, FP, FN
- **Loss Curve**: Grafik penurunan loss per epoch

---

## 🌐 Fitur Web App

| Halaman | URL | Deskripsi |
|---------|-----|-----------|
| Form Prediksi | `/` | Input data mahasiswa & pilih model |
| Hasil Prediksi | `/predict` | Tampilan hasil 5 model + voting |
| Perbandingan | `/compare` | Grafik & tabel perbandingan model |
| EDA | `/eda` | Visualisasi eksplorasi data |
| Tentang | `/about` | Deskripsi proyek & algoritma |
| API | `/api/predict` | REST API (JSON) |

---

## 🚀 Deploy ke Railway / Render

### Railway
1. Push ke GitHub
2. Buat project baru di [railway.app](https://railway.app)
3. Connect repository → Railway otomatis membaca `Procfile`
4. Tambahkan environment variable jika diperlukan

### Render
1. Push ke GitHub
2. Buat **Web Service** baru di [render.com](https://render.com)
3. Build command: `pip install -r requirements.txt && python notebooks/train_models.py`
4. Start command: `gunicorn app.app:app`

---

## 📦 Teknologi

- **Python 3.9+**
- **scikit-learn** — Model ML
- **Flask** — Web framework
- **Bootstrap 5** — Frontend UI
- **Pandas, NumPy** — Data processing
- **Matplotlib, Seaborn** — Visualisasi
- **Joblib** — Serialisasi model

---

## 👤 Informasi Proyek

- **Mata Kuliah**: Praktikum Kecerdasan Buatan
- **Jenis**: Ujian Tengah Semester (UTS)
- **Dataset**: Student Performance Prediction — Kaggle
- **Tahun**: 2025
