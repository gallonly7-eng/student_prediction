# LAPORAN JURNAL ILMIAH
## Prediksi Kelulusan Mahasiswa Menggunakan Algoritma Machine Learning

---

**Judul**: Implementasi dan Perbandingan Lima Algoritma Machine Learning
untuk Prediksi Kelulusan Mahasiswa Berdasarkan Data Akademik

**Mata Kuliah**: Praktikum Kecerdasan Buatan

---

## ABSTRAK

Penelitian ini mengimplementasikan dan membandingkan lima algoritma machine learning
untuk memprediksi kelulusan mahasiswa berdasarkan data akademik. Dataset yang digunakan
adalah Student Performance Prediction Dataset dari Kaggle yang berisi 40.000 data mahasiswa
dengan fitur: jam belajar per minggu, tingkat kehadiran, nilai sebelumnya, keikutsertaan
ekstrakulikuler, dan tingkat pendidikan orang tua.

Lima algoritma yang diimplementasikan adalah: (1) Logistic Regression sebagai baseline
model, (2) Artificial Neural Network (ANN) dengan arsitektur MLP 2 hidden layer,
(3) RNN/LSTM yang disimulasikan dengan Deep Sequential MLP 5 hidden layer menggunakan
aktivasi tanh, (4) K-Means Clustering sebagai pendekatan unsupervised learning, dan
(5) Model Backpropagation dengan SGD optimizer dan fungsi aktivasi sigmoid.

Hasil eksperimen menunjukkan bahwa model Backpropagation mencapai akurasi tertinggi
sebesar 99.91% diikuti Logistic Regression (99.88%), ANN (99.75%), RNN/LSTM (99.61%),
dan K-Means (50.85%). K-Means memiliki keterbatasan karena bersifat unsupervised sehingga
tidak dapat langsung memetakan cluster ke label kelulusan secara optimal.

**Kata kunci**: machine learning, prediksi kelulusan, logistic regression, neural network,
backpropagation, k-means clustering

---

## 1. PENDAHULUAN

### 1.1 Latar Belakang

Kelulusan mahasiswa merupakan indikator penting dalam menilai kualitas proses pendidikan
di perguruan tinggi. Kemampuan untuk memprediksi kemungkinan kelulusan mahasiswa secara
dini dapat membantu institusi pendidikan mengambil tindakan preventif guna meningkatkan
tingkat kelulusan. Machine learning menawarkan solusi yang efektif dalam mengidentifikasi
pola dari data historis akademik mahasiswa untuk melakukan prediksi tersebut.

Beberapa penelitian sebelumnya telah menggunakan berbagai algoritma machine learning
seperti Decision Tree, Random Forest, dan SVM untuk prediksi dropout mahasiswa. Namun,
perbandingan komprehensif antara metode tradisional (Logistic Regression), neural network
(ANN, RNN/LSTM), unsupervised (K-Means), dan model backpropagation dalam satu framework
masih jarang dilakukan.

### 1.2 Rumusan Masalah

1. Algoritma machine learning mana yang paling efektif untuk memprediksi kelulusan mahasiswa?
2. Bagaimana performa komparatif antara model supervised dan unsupervised dalam konteks ini?
3. Fitur akademik apa yang paling berpengaruh terhadap kelulusan mahasiswa?

### 1.3 Tujuan

1. Mengimplementasikan 5 algoritma ML: Logistic Regression, ANN, RNN/LSTM, K-Means, Backpropagation
2. Membandingkan performa berdasarkan metrik Accuracy, MAE, RMSE, dan R² Score
3. Membangun aplikasi web prediksi menggunakan Flask

---

## 2. TINJAUAN PUSTAKA

### 2.1 Logistic Regression

Logistic Regression adalah algoritma klasifikasi yang memodelkan probabilitas kelas
menggunakan fungsi sigmoid (logistic). Untuk klasifikasi biner seperti lulus/tidak lulus,
output model adalah probabilitas P(y=1|X) yang kemudian di-threshold pada 0.5.

```
P(y=1|X) = 1 / (1 + e^(-θᵀX))
```

### 2.2 Artificial Neural Network (ANN)

ANN meniru cara kerja neuron biologis. Dalam implementasi ini digunakan MLPClassifier
dengan arsitektur:
- Input layer: 5 neuron (sesuai jumlah fitur)
- Hidden layer 1: 64 neuron, aktivasi ReLU
- Hidden layer 2: 32 neuron, aktivasi ReLU
- Output layer: 1 neuron (sigmoid), 2 kelas

Optimizer: Adam dengan early stopping untuk mencegah overfitting.

### 2.3 RNN/LSTM

Recurrent Neural Network dirancang untuk data sekuensial. Karena keterbatasan library
(tidak ada TensorFlow), RNN/LSTM disimulasikan menggunakan Deep MLP dengan:
- 5 hidden layer: (128, 64, 64, 32, 16)
- Aktivasi **tanh** — sama seperti fungsi aktivasi gating di LSTM
- Adaptive learning rate untuk meniru perilaku adaptif RNN

### 2.4 K-Means Clustering

K-Means adalah algoritma clustering unsupervised yang mempartisi data ke dalam K cluster
dengan meminimalkan Within-Cluster Sum of Squares (WCSS):

```
WCSS = Σ Σ ||xᵢ - μⱼ||²
```

Untuk k=2, cluster disesuaikan dengan label aktual menggunakan majority voting.

### 2.5 Backpropagation

Backpropagation adalah algoritma pelatihan jaringan saraf yang menghitung gradien
menggunakan chain rule untuk memperbarui bobot:

```
Δw = -η × ∂L/∂w
```

Implementasi menggunakan SGD optimizer dengan momentum 0.9 dan learning rate 0.01.

---

## 3. METODOLOGI

### 3.1 Dataset

| Atribut | Nilai |
|---------|-------|
| Sumber | Kaggle — Student Performance Prediction |
| Jumlah sampel | 40.000 |
| Jumlah fitur | 5 (setelah preprocessing) |
| Kelas target | 2 (Lulus / Tidak Lulus) |
| Missing values | ~5% per kolom |
| Distribusi kelas | Seimbang (50:50) |

**Fitur yang digunakan:**
1. Study Hours per Week (jam belajar per minggu)
2. Attendance Rate (tingkat kehadiran %)
3. Previous Grades (nilai akademik sebelumnya)
4. Participation in Extracurricular Activities (0/1)
5. Parent Education Level (ordinal 0-4)

### 3.2 Preprocessing

**Langkah-langkah preprocessing:**

1. **Handling Missing Values**
   - Kolom numerik: imputasi dengan nilai median (robust terhadap outlier)
   - Kolom kategorikal: imputasi dengan nilai modus

2. **Handling Outlier**
   - Metode IQR (Interquartile Range) 1.5× untuk clipping nilai ekstrem

3. **Encoding Kategorikal**
   - Participation: binary mapping (Yes=1, No=0)
   - Parent Education Level: ordinal encoding (High School=0, ..., PhD=4)

4. **Feature Engineering Target**
   - Skor akademik tertimbang:
     `Score = (Previous Grades × 0.50) + (Attendance Rate × 0.35) + (Study Hours × 1.50)`
   - Threshold: median skor = 73.86
   - Mahasiswa lulus jika Score ≥ threshold

5. **Train-Test Split**: 80% training, 20% testing (stratified)

6. **Standardisasi**: StandardScaler (mean=0, std=1)

### 3.3 Evaluasi Model

Metrik yang digunakan:

| Metrik | Formula | Keterangan |
|--------|---------|------------|
| Accuracy | (TP+TN)/(TP+TN+FP+FN) | % prediksi benar |
| MAE | Σ\|yᵢ-ŷᵢ\|/n | Rata-rata error absolut |
| RMSE | √(Σ(yᵢ-ŷᵢ)²/n) | Root mean square error |
| R² Score | 1 - SS_res/SS_tot | Koefisien determinasi |
| Confusion Matrix | [TN FP; FN TP] | Matriks klasifikasi |

---

## 4. HASIL DAN PEMBAHASAN

### 4.1 EDA (Exploratory Data Analysis)

**Statistik Deskriptif Fitur Utama:**

| Fitur | Mean | Std | Min | Max |
|-------|------|-----|-----|-----|
| Study Hours/Week | 10.02 | 5.01 | 0 | 25 |
| Attendance Rate | 75.08 | 20.0 | 0 | 100 |
| Previous Grades | 65.45 | 16.5 | 0 | 100 |

**Temuan EDA:**
- Distribusi ketiga fitur numerik mendekati normal (bell-shaped)
- Korelasi tertinggi dengan target: Study Hours (0.48), Previous Grades (0.47), Attendance (0.44)
- Fitur ekstrakulikuler dan pendidikan orang tua memiliki korelasi kecil (<0.1)
- Dataset seimbang: 50% lulus, 50% tidak lulus (setelah rekonstruksi target)

### 4.2 Hasil Evaluasi Model

| Model | Accuracy | MAE | RMSE | R² Score |
|-------|----------|-----|------|----------|
| Logistic Regression | 99.88% | 0.0013 | 0.0354 | 0.9782 |
| ANN (MLP) | 99.75% | 0.0025 | 0.0500 | 0.9922 |
| RNN/LSTM (Deep MLP) | 99.61% | 0.0039 | 0.0622 | 0.9894 |
| K-Means Clustering | 50.85% | 0.4915 | 0.7011 | -0.9660 |
| **Backpropagation** | **99.91%** | **0.0009** | **0.0296** | **0.9795** |

### 4.3 Analisis Per Model

**Logistic Regression (99.88%)**
Performa sangat tinggi menunjukkan bahwa hubungan antara fitur dan target bersifat
linear yang kuat. MAE rendah (0.0013) berarti hampir tidak ada kesalahan prediksi.

**ANN (MLP) (99.75%)**
Jaringan 2 layer mampu mempelajari pola non-linear dengan baik. R² = 0.9922 tertinggi
di antara semua model menunjukkan probabilitas prediksi sangat akurat.

**RNN/LSTM (99.61%)**
Arsitektur paling dalam (5 layer) dengan aktivasi tanh. Akurasi sedikit lebih rendah
karena kompleksitas model yang lebih tinggi, namun masih sangat baik.

**K-Means Clustering (50.85%)**
Performa rendah karena merupakan metode unsupervised yang tidak bisa belajar dari label.
R² negatif menunjukkan probabilitas output tidak terkorelasi dengan label aktual.
Silhouette Score 0.22 menunjukkan cluster cukup terpisah secara geometris.

**Backpropagation (99.91%) — TERBAIK**
Model terbaik dengan RMSE terendah (0.0296) dan akurasi tertinggi. SGD dengan momentum
memberikan konvergensi yang stabil dan generalisasi yang baik.

### 4.4 Analisis Fitur Penting

Berdasarkan korelasi Pearson dengan target kelulusan:
1. Study Hours per Week: 0.478 (pengaruh terbesar)
2. Previous Grades: 0.467
3. Attendance Rate: 0.440
4. Parent Education Level: 0.090
5. Participation in Extracurricular: 0.080

---

## 5. KESIMPULAN

### 5.1 Kesimpulan Utama

1. **Model Backpropagation (SGD)** adalah model terbaik dengan akurasi **99.91%**,
   MAE 0.0009, RMSE 0.0296, dan R² 0.9795 pada data uji.

2. Algoritma supervised learning (Logistic Regression, ANN, Backpropagation) jauh
   mengungguli K-Means karena dapat belajar dari label pelatihan.

3. Fitur **jam belajar per minggu**, **nilai sebelumnya**, dan **tingkat kehadiran**
   adalah tiga fitur paling berpengaruh terhadap prediksi kelulusan.

4. Aplikasi web berbasis Flask berhasil dibangun dengan 5 halaman: form prediksi,
   hasil prediksi, perbandingan model, EDA, dan halaman tentang.

### 5.2 Saran

1. Menggunakan dataset nyata dari institusi pendidikan untuk validasi model
2. Menambahkan fitur seperti IPK semester sebelumnya, beban SKS, dan keaktifan organisasi
3. Mengimplementasikan TensorFlow/Keras untuk RNN/LSTM sesungguhnya
4. Menambahkan SHAP values untuk explainability model

---

## DAFTAR PUSTAKA

1. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825-2830.
2. Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735-1780.
3. Goodfellow, I., Bengio, Y., & Courville, A. (2016). *Deep Learning*. MIT Press.
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning*. Springer.
5. Romero, C., & Ventura, S. (2010). Educational data mining: A review. *Expert Systems with Applications*, 37(12).
6. Kaggle. (2024). Student Performance Prediction Dataset. https://www.kaggle.com
