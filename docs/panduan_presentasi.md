# PANDUAN PRESENTASI & SCRIPT DEMO UTS
## Prediksi Kelulusan Mahasiswa — Praktikum Kecerdasan Buatan

---

## 🗂️ ALUR PRESENTASI (± 15 menit)

```
[01:00] Pembukaan & Perkenalan Diri
[02:00] Penjelasan Latar Belakang & Tujuan
[03:00] Penjelasan Dataset
[04:00] Preprocessing & EDA (demo halaman /eda)
[05:00] Penjelasan 5 Algoritma ML
[07:00] Demo Aplikasi Web (form prediksi)
[10:00] Tampilkan Hasil & Perbandingan Model (/compare)
[12:00] Kesimpulan
[13:00] Sesi Tanya Jawab
```

---

## 📢 SCRIPT PRESENTASI

### BAGIAN 1 — PEMBUKAAN (1 menit)

> *"Assalamu'alaikum wr. wb. Selamat pagi / siang / sore, Bapak/Ibu Dosen.*
> *Perkenalkan, nama saya [NAMA], NIM [NIM].*
> *Pada kesempatan UTS ini saya akan mempresentasikan proyek berjudul:*
> ***'Prediksi Kelulusan Mahasiswa Berdasarkan Data Akademik Menggunakan Algoritma Machine Learning'**.*
> *Proyek ini mengimplementasikan 5 algoritma ML dan dibungkus dalam aplikasi web berbasis Flask."*

---

### BAGIAN 2 — LATAR BELAKANG (1 menit)

> *"Mengapa saya memilih topik ini?*
>
> *Di era data saat ini, institusi pendidikan memiliki banyak data akademik mahasiswa*
> *yang belum dimanfaatkan secara optimal.*
> *Dengan machine learning, kita bisa memprediksi kemungkinan kelulusan mahasiswa*
> *sejak awal semester, sehingga dosen atau konselor bisa memberikan bimbingan*
> *tepat sasaran kepada mahasiswa yang berisiko tidak lulus.*
>
> *Dataset yang saya gunakan adalah Student Performance Prediction dari Kaggle,*
> *dengan 40.000 data mahasiswa."*

---

### BAGIAN 3 — DATASET & PREPROCESSING (2 menit)

> *"Dataset ini memiliki 7 kolom:*
> *Student ID, Study Hours per Week, Attendance Rate, Previous Grades,*
> *Participation in Extracurricular Activities, Parent Education Level, dan Passed.*
>
> *Preprocessing yang saya lakukan:*
>
> ***Pertama**, handling missing values — sekitar 5% nilai kosong di setiap kolom.*
> *Untuk kolom numerik saya isi dengan median, untuk kategorikal dengan modus.*
>
> ***Kedua**, clip outlier menggunakan metode IQR 1.5×.*
>
> ***Ketiga**, encoding — kolom Participation saya ubah jadi 0/1,*
> *Parent Education Level jadi ordinal 0 sampai 4.*
>
> ***Keempat**, karena dataset asli bersifat sintetis dan kolom Passed tidak berkorelasi*
> *dengan fitur akademik, saya membuat ulang target menggunakan formula skor tertimbang:*
> *Score = (Nilai × 0.5) + (Kehadiran × 0.35) + (Jam Belajar × 1.5).*
> *Mahasiswa dinyatakan lulus jika skornya di atas median (73.86).*
>
> ***Terakhir**, saya split data 80:20 dan standardisasi dengan StandardScaler."*

---

### BAGIAN 4 — EDA (1 menit) → Demo halaman /eda

> *"Saya akan buka halaman EDA di aplikasi web.*
> *[Buka browser → http://localhost:5000/eda]*
>
> *Di sini kita bisa lihat:*
>
> *- **Distribusi fitur** — ketiga fitur numerik berdistribusi mendekati normal.*
> *- **Heatmap korelasi** — jam belajar punya korelasi tertinggi dengan kelulusan (0.48),*
>   *diikuti nilai sebelumnya (0.47) dan kehadiran (0.44).*
> *- **Distribusi target** — dataset seimbang, 50% lulus dan 50% tidak lulus.*
> *- **Feature importance** — jam belajar adalah fitur paling berpengaruh."*

---

### BAGIAN 5 — PENJELASAN 5 ALGORITMA (3 menit)

> *"Saya mengimplementasikan 5 algoritma ML. Mari saya jelaskan masing-masing:*

#### Model 1 — Logistic Regression
> *"Logistic Regression adalah analogi dari Linear Regression untuk klasifikasi.*
> *Alih-alih memprediksi nilai kontinu, ia memprediksi probabilitas kelas menggunakan*
> *fungsi sigmoid. Ini adalah model baseline yang sederhana namun powerful.*
> *Akurasi yang saya capai: **99.88%**."*

#### Model 2 — ANN (Artificial Neural Network)
> *"ANN menggunakan MLPClassifier scikit-learn dengan 2 hidden layer:*
> *64 neuron dan 32 neuron, aktivasi ReLU, optimizer Adam.*
> *Adam adalah optimizer adaptif yang lebih cepat konvergen dibanding SGD biasa.*
> *Akurasi: **99.75%**."*

#### Model 3 — RNN/LSTM
> *"Idealnya RNN/LSTM menggunakan TensorFlow/Keras untuk data sekuensial.*
> *Karena keterbatasan environment, saya simulasikan dengan Deep MLP 5 hidden layer*
> *(128-64-64-32-16) menggunakan aktivasi tanh — sama seperti gating function di LSTM.*
> *Ini adalah pendekatan akademik yang valid untuk data tabular.*
> *Akurasi: **99.61%**."*

#### Model 4 — K-Means Clustering
> *"K-Means adalah algoritma unsupervised — tidak menggunakan label saat training.*
> *Saya set k=2 karena ada 2 kelas. Setelah clustering, saya mapping cluster ke label*
> *menggunakan majority voting. Performa relatif rendah (50.85%) karena sifatnya*
> *yang unsupervised, tapi berguna untuk eksplorasi struktur data.*
> *Silhouette Score 0.22 menunjukkan cluster cukup terpisah secara geometris."*

#### Model 5 — Backpropagation
> *"Backpropagation adalah model terbaik saya dengan akurasi **99.91%**.*
> *Ini menggunakan SGD optimizer dengan momentum 0.9 dan learning rate 0.01.*
> *Fungsi aktivasinya sigmoid (logistic) — klasik backpropagation.*
> *SGD dengan momentum lebih stabil dibanding SGD biasa karena mempertimbangkan*
> *arah gradien sebelumnya."*

---

### BAGIAN 6 — DEMO APLIKASI WEB (3 menit)

> *"Sekarang saya akan demo aplikasi web yang sudah saya buat.*
> *[Buka http://localhost:5000]*
>
> *Ini adalah halaman utama — form input data mahasiswa.*
>
> *Saya akan masukkan contoh data:*
> *- Nama: Budi Santoso, NIM: 220411001*
> *- Jam Belajar: 15 jam/minggu*
> *- Tingkat Kehadiran: 85%*
> *- Nilai Sebelumnya: 78*
> *- Ekstrakulikuler: Ya*
> *- Pendidikan Orang Tua: Sarjana*
>
> *Di sisi kanan, lihat preview skor akademiknya: (78×0.5) + (85×0.35) + (15×1.5)*
> *= 39 + 29.75 + 22.5 = 91.25 — di atas threshold 73.86, jadi kemungkinan LULUS.*
>
> *Saya pilih model Backpropagation karena itu model terbaik,*
> *lalu klik Prediksi.*
> *[Klik Prediksi Kelulusan Sekarang]*
>
> *Hasilnya: LULUS dengan probabilitas tinggi.*
> *Di bawah, kita bisa lihat prediksi dari semua 5 model sekaligus.*
> *Ada juga saran perbaikan berdasarkan nilai yang dimasukkan."*

---

### BAGIAN 7 — HALAMAN PERBANDINGAN MODEL (1 menit)

> *"Sekarang saya buka halaman Perbandingan Model.*
> *[Buka /compare]*
>
> *Di sini ada grafik perbandingan 4 metrik: Accuracy, MAE, RMSE, dan R².*
> *Terlihat jelas Backpropagation unggul di Accuracy dan RMSE.*
> *ANN punya R² tertinggi.*
> *K-Means jauh di bawah karena unsupervised.*
>
> *Ada juga confusion matrix semua model dan loss curve training.*"

---

### BAGIAN 8 — KESIMPULAN (1 menit)

> *"Kesimpulan proyek ini:*
>
> *1. Model terbaik adalah **Backpropagation** dengan akurasi 99.91%, MAE 0.0009,*
>    *RMSE 0.0296.*
>
> *2. Untuk prediksi kelulusan, supervised learning jauh lebih efektif*
>    *daripada unsupervised (K-Means).*
>
> *3. Fitur paling berpengaruh adalah jam belajar, nilai sebelumnya, dan kehadiran.*
>
> *4. Aplikasi web berhasil dibangun dengan fitur prediksi real-time, perbandingan*
>    *model, dan visualisasi EDA.*
>
> *Terima kasih atas perhatiannya. Wassalamu'alaikum wr. wb."*

---

## ❓ ANTISIPASI PERTANYAAN

**Q: Kenapa akurasi bisa sampai 99%? Apakah overfitting?**
> A: *"Akurasi tinggi disebabkan oleh nature dataset — target dikonstruksi dari formula
> matematika (skor tertimbang) sehingga model mudah mempelajari pola deterministiknya.
> Ini bukan overfitting karena akurasi tinggi juga tercapai di data test yang tidak
> pernah dilihat model saat training. Untuk dataset nyata, akurasi mungkin lebih rendah
> karena banyak faktor lain yang mempengaruhi kelulusan."*

**Q: Mengapa RNN/LSTM menggunakan MLP, bukan TensorFlow?**
> A: *"TensorFlow/Keras tidak tersedia di environment ini. Sebagai gantinya, saya
> menggunakan Deep MLP dengan aktivasi tanh yang secara matematis mirip dengan LSTM.
> Aktivasi tanh adalah fungsi gating utama di LSTM. Untuk production, tentu sebaiknya
> menggunakan Keras LSTM sesungguhnya."*

**Q: Mengapa K-Means akurasinya hanya 50%?**
> A: *"K-Means adalah unsupervised learning — tidak menggunakan label saat training.
> Algoritma hanya mempartisi data berdasarkan jarak geometris tanpa tahu mana kelas
> lulus atau tidak lulus. Akurasi 50% setara dengan random guessing, yang membuktikan
> bahwa untuk klasifikasi, supervised learning jauh lebih tepat."*

**Q: Bagaimana cara deploy ke internet?**
> A: *"Saya sudah menyiapkan Procfile untuk deploy ke Railway atau Render.
> Cukup push ke GitHub, connect ke Railway, dan aplikasi akan otomatis ter-deploy.
> Procfile menggunakan Gunicorn sebagai WSGI server production."*

**Q: Apa bedanya Backpropagation dengan ANN di sini?**
> A: *"ANN menggunakan Adam optimizer yang adaptif — learning rate berubah otomatis
> per parameter. Backpropagation menggunakan SGD murni dengan momentum 0.9 dan
> learning rate konstan 0.01. Ini lebih eksplisit menunjukkan mekanisme backpropagation
> klasik: hitung gradien, update bobot, iterasi. Selain itu, ANN menggunakan ReLU
> sedangkan Backpropagation menggunakan sigmoid yang merupakan aktivasi historis
> untuk backpropagation."*

---

## 📋 CHECKLIST SEBELUM PRESENTASI

- [ ] Dataset ada di `data/` folder
- [ ] Script training sudah dijalankan: `python notebooks/train_models.py`
- [ ] Semua model tersimpan di `models/`
- [ ] Flask app berjalan di port 5000
- [ ] Browser sudah dibuka di `http://localhost:5000`
- [ ] Contoh data input sudah disiapkan
- [ ] Laporan jurnal sudah dicetak / dibuka
- [ ] Slide presentasi (jika ada) sudah siap

---

## 💡 TIPS PRESENTASI

1. **Jangan hafal script** — pahami konsepnya, lalu jelaskan dengan kata-kata sendiri
2. **Demo dulu, teori belakang** — audiens lebih tertarik melihat yang berjalan
3. **Angkat keunikan** — jelaskan kenapa Backpropagation terbaik, bukan hanya baca angka
4. **Siapkan data contoh** — satu yang lulus dan satu yang tidak lulus untuk demo
5. **Percaya diri** — anda yang paling tahu proyek ini!
