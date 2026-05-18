"""
=========================================================
  PROYEK UTS: Prediksi Kelulusan Mahasiswa
  File   : app/app.py
  Fungsi : Flask Web Application
=========================================================
  Routes:
    /           → Halaman utama (form input mahasiswa)
    /predict    → Proses prediksi (POST)
    /compare    → Halaman perbandingan model
    /eda        → Halaman EDA & visualisasi
    /about      → Tentang proyek
=========================================================
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib
import json
import os

# ── Inisialisasi Flask ────────────────────────────────────────────
app = Flask(__name__)

# ── Path ke model & metadata ──────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

# ── Load semua model saat aplikasi startup ────────────────────────
print("[Flask] Memuat model...")

scaler    = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
lr_model  = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl"))
ann_model = joblib.load(os.path.join(MODEL_DIR, "ann_model.pkl"))
rnn_model = joblib.load(os.path.join(MODEL_DIR, "rnn_model.pkl"))
km_model  = joblib.load(os.path.join(MODEL_DIR, "kmeans_model.pkl"))
bp_model  = joblib.load(os.path.join(MODEL_DIR, "backpropagation.pkl"))
pca_model = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))

with open(os.path.join(MODEL_DIR, "results.json")) as f:
    model_results = json.load(f)

with open(os.path.join(MODEL_DIR, "meta.json")) as f:
    meta = json.load(f)

with open(os.path.join(MODEL_DIR, "prep_meta.json")) as f:
    prep_meta = json.load(f)

THRESHOLD = prep_meta["threshold"]

print(f"[Flask] Semua model dimuat. Model terbaik: {meta['best_model']}")

# ── Daftar model untuk UI ──────────────────────────────────────────
MODELS = {
    "logistic":  ("Logistic Regression",    lr_model,  "#4e8fde"),
    "ann":       ("ANN (MLP)",              ann_model, "#4caf50"),
    "rnn":       ("RNN/LSTM (Deep MLP)",    rnn_model, "#9c27b0"),
    "kmeans":    ("K-Means Clustering",     km_model,  "#ff9800"),
    "backprop":  ("Backpropagation",        bp_model,  "#e91e63"),
}

EDUCATION_MAP = {
    "High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "PhD": 4
}


def preprocess_input(study_hours, attendance, prev_grades, participation, parent_edu):
    """
    Preprocessing input dari form web:
    1. Encode variabel kategorikal ke numerik
    2. Buat vektor fitur
    3. Standardisasi dengan scaler yang sudah di-fit saat training
    """
    participation_enc = 1 if participation == "Yes" else 0
    parent_edu_enc    = EDUCATION_MAP.get(parent_edu, 2)

    features = np.array([[
        float(study_hours),
        float(attendance),
        float(prev_grades),
        float(participation_enc),
        float(parent_edu_enc)
    ]])

    features_scaled = scaler.transform(features)
    return features, features_scaled


def predict_all_models(features_scaled):
    """
    Jalankan prediksi untuk semua 5 model sekaligus.
    Return: dict berisi prediksi & probabilitas tiap model.
    """
    predictions = {}

    for key, (name, model, color) in MODELS.items():
        if key == "kmeans":
            # K-Means: predict cluster, lalu map ke label
            cluster = model.predict(features_scaled)[0]
            # Heuristik: cluster 0/1 → label berdasarkan centroid
            centroids = model.cluster_centers_
            # Centroid yang lebih tinggi → lebih mungkin lulus
            score_0   = float(np.mean(centroids[0]))
            score_1   = float(np.mean(centroids[1]))
            if score_0 > score_1:
                pred  = 1 if cluster == 0 else 0
            else:
                pred  = 1 if cluster == 1 else 0
            prob = 0.75 if pred == 1 else 0.25
        else:
            pred = int(model.predict(features_scaled)[0])
            prob = float(model.predict_proba(features_scaled)[0][1])

        predictions[key] = {
            "name":       name,
            "color":      color,
            "prediction": pred,
            "label":      "LULUS" if pred == 1 else "TIDAK LULUS",
            "probability": round(prob * 100, 1),
            "confidence": "Tinggi" if abs(prob - 0.5) > 0.3 else "Sedang"
        }

    return predictions


# ══════════════════════════════════════════════════════════════════
# ROUTES
# ══════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Halaman utama — form input data mahasiswa"""
    return render_template("index.html",
                           best_model=meta["best_model"],
                           best_accuracy=meta["best_accuracy"])


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint prediksi:
    1. Ambil data dari form
    2. Preprocessing
    3. Prediksi dengan semua model
    4. Render halaman hasil
    """
    try:
        # Ambil data dari form
        nama         = request.form.get("nama", "Mahasiswa")
        nim          = request.form.get("nim", "-")
        study_hours  = float(request.form.get("study_hours",  10))
        attendance   = float(request.form.get("attendance",   75))
        prev_grades  = float(request.form.get("prev_grades",  65))
        participation = request.form.get("participation", "No")
        parent_edu   = request.form.get("parent_edu",    "Bachelor")
        selected_model = request.form.get("selected_model", "backprop")

        # Validasi range input
        study_hours = max(0, min(40,  study_hours))
        attendance  = max(0, min(100, attendance))
        prev_grades = max(0, min(100, prev_grades))

        # Preprocessing
        features_raw, features_scaled = preprocess_input(
            study_hours, attendance, prev_grades, participation, parent_edu)

        # Prediksi semua model
        all_predictions = predict_all_models(features_scaled)

        # Hitung skor akademik (untuk informasi)
        academic_score = (prev_grades * 0.50 +
                          attendance  * 0.35 +
                          study_hours * 1.50)

        # Voting majority dari semua model (kecuali K-Means)
        votes   = [all_predictions[k]["prediction"]
                   for k in all_predictions if k != "kmeans"]
        majority = 1 if sum(votes) > len(votes)/2 else 0

        input_data = {
            "nama":          nama,
            "nim":           nim,
            "study_hours":   study_hours,
            "attendance":    attendance,
            "prev_grades":   prev_grades,
            "participation": participation,
            "parent_edu":    parent_edu,
            "academic_score": round(academic_score, 2),
            "threshold":     round(THRESHOLD, 2)
        }

        return render_template("result.html",
                               input_data=input_data,
                               all_predictions=all_predictions,
                               selected_model=selected_model,
                               majority_vote=majority,
                               majority_label="LULUS" if majority == 1 else "TIDAK LULUS")

    except Exception as e:
        return render_template("index.html",
                               error=f"Error: {str(e)}",
                               best_model=meta["best_model"],
                               best_accuracy=meta["best_accuracy"])


@app.route("/compare")
def compare():
    """Halaman perbandingan performa semua model"""
    # Hitung ranking berdasarkan accuracy
    ranked = sorted(model_results.items(),
                    key=lambda x: x[1]["Accuracy"], reverse=True)

    return render_template("compare.html",
                           results=model_results,
                           ranked=ranked,
                           best_model=meta["best_model"],
                           meta=meta)


@app.route("/eda")
def eda():
    """Halaman EDA & visualisasi dataset"""
    return render_template("eda.html",
                           meta=meta)


@app.route("/about")
def about():
    """Halaman tentang proyek"""
    return render_template("about.html",
                           meta=meta)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint untuk prediksi (JSON).
    Berguna untuk integrasi dengan sistem lain.
    """
    try:
        data = request.get_json()
        features_raw, features_scaled = preprocess_input(
            data.get("study_hours",  10),
            data.get("attendance",   75),
            data.get("prev_grades",  65),
            data.get("participation","No"),
            data.get("parent_edu",  "Bachelor")
        )
        predictions = predict_all_models(features_scaled)
        return jsonify({"status": "success", "predictions": predictions})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400


# ── Entry point ───────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
