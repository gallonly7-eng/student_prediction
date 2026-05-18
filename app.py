"""
=========================================================
  PROYEK UTS: Prediksi Kelulusan Mahasiswa
  File   : app.py  (file utama — cukup jalankan ini)
  Cara   : python app.py
=========================================================
  Alur otomatis saat dijalankan:
    1. Cek apakah model sudah ada di folder models/
    2. Jika BELUM → training otomatis (± 1-2 menit)
    3. Jika SUDAH  → langsung load model
    4. Jalankan Flask di http://localhost:5000
=========================================================
"""

import os
import sys
import json
import warnings
import webbrowser
import threading

warnings.filterwarnings("ignore")

# ── Tentukan BASE_DIR (folder tempat app.py berada) ───────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data")
IMG_DIR    = os.path.join(BASE_DIR, "app", "static", "img")
DATA_PATH  = os.path.join(DATA_DIR, "student_performance_prediction.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR,   exist_ok=True)


# ══════════════════════════════════════════════════════════════════
#  FUNGSI TRAINING — dipanggil otomatis jika model belum ada
# ══════════════════════════════════════════════════════════════════
def run_training():
    """
    Training semua 5 model ML + simpan grafik EDA.
    Dipanggil otomatis saat model belum ada di folder models/.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    from sklearn.model_selection  import train_test_split
    from sklearn.preprocessing    import StandardScaler
    from sklearn.linear_model     import LogisticRegression
    from sklearn.neural_network   import MLPClassifier
    from sklearn.cluster          import KMeans
    from sklearn.decomposition    import PCA
    from sklearn.metrics          import (accuracy_score, mean_absolute_error,
                                           mean_squared_error, r2_score,
                                           confusion_matrix, silhouette_score)
    from scipy.stats import mode
    import joblib

    print("=" * 60)
    print("  AUTO TRAINING — model belum ditemukan, mulai training...")
    print("=" * 60)

    # ── 1. LOAD DATA ─────────────────────────────────────────────
    print("\n[1/7] Load dataset...")
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["Student ID"], errors="ignore")
    print(f"  → {df.shape[0]:,} baris × {df.shape[1]} kolom")

    # ── 2. PREPROCESSING ──────────────────────────────────────────
    print("\n[2/7] Preprocessing...")
    num_cols = ["Study Hours per Week", "Attendance Rate", "Previous Grades"]
    cat_cols = ["Participation in Extracurricular Activities", "Parent Education Level"]

    df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna(df[c].mode()[0])
    df.dropna(subset=["Passed"], inplace=True)

    # Clip outlier IQR 1.5
    for c in num_cols:
        Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
        df[c]  = df[c].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

    # Encoding
    df["Participation_Enc"] = df["Participation in Extracurricular Activities"]\
                               .map({"Yes":1,"No":0}).fillna(0).astype(int)
    edu_map = {"High School":0,"Associate":1,"Bachelor":2,"Master":3,"PhD":4}
    df["ParentEdu_Enc"] = df["Parent Education Level"].map(edu_map).fillna(2).astype(int)

    # Target: skor akademik tertimbang → threshold median
    df["AcademicScore"] = (df["Previous Grades"]       * 0.50 +
                            df["Attendance Rate"]        * 0.35 +
                            df["Study Hours per Week"]   * 1.50)
    threshold = df["AcademicScore"].median()
    df["Passed_Binary"] = (df["AcademicScore"] >= threshold).astype(int)

    FEATURES = ["Study Hours per Week","Attendance Rate",
                "Previous Grades","Participation_Enc","ParentEdu_Enc"]
    TARGET   = "Passed_Binary"

    df.dropna(subset=FEATURES + [TARGET], inplace=True)
    X = df[FEATURES].values.astype(float)
    y = df[TARGET].values.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    with open(os.path.join(MODEL_DIR, "prep_meta.json"), "w") as f:
        json.dump({"threshold": float(threshold), "features": FEATURES}, f)

    print(f"  → Train: {len(X_train):,} | Test: {len(X_test):,} | Threshold: {threshold:.2f}")

    # ── 3. EDA ───────────────────────────────────────────────────
    print("\n[3/7] EDA & grafik...")

    # Distribusi fitur
    fig, axes = plt.subplots(1,3,figsize=(15,4))
    fig.suptitle("Distribusi Fitur Numerik Utama", fontsize=14, fontweight="bold")
    for ax, col, color in zip(axes,
        ["Study Hours per Week","Attendance Rate","Previous Grades"],
        ["#4e8fde","#4caf50","#ff9800"]):
        sns.histplot(df[col], kde=True, ax=ax, color=color, bins=40)
        ax.set_title(col, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,"dist_fitur.png"), dpi=110, bbox_inches="tight"); plt.close()

    # Heatmap korelasi
    corr = df[FEATURES+["Passed_Binary"]].corr()
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Heatmap Korelasi Antar Fitur", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,"heatmap_korelasi.png"), dpi=110, bbox_inches="tight"); plt.close()

    # Distribusi target
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    fig.suptitle("Analisis Distribusi Kelulusan", fontsize=13, fontweight="bold")
    vc = df["Passed_Binary"].value_counts().sort_index()
    axes[0].pie([vc.get(0,0), vc.get(1,0)], labels=["Tidak Lulus","Lulus"],
                autopct="%1.1f%%", colors=["#f44336","#4caf50"], startangle=90)
    axes[0].set_title("Proporsi Kelulusan")
    sns.boxplot(x="Passed_Binary", y="Previous Grades", data=df,
                palette={0:"#f44336",1:"#4caf50"}, ax=axes[1])
    axes[1].set_title("Previous Grades vs Kelulusan")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,"target_dist.png"), dpi=110, bbox_inches="tight"); plt.close()

    # Feature importance
    fi = corr["Passed_Binary"].drop("Passed_Binary").abs().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(7,4))
    fi.plot(kind="barh", ax=ax, color="#4e8fde")
    ax.set_title("Korelasi Fitur dengan Kelulusan (|nilai|)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,"feature_importance.png"), dpi=110, bbox_inches="tight"); plt.close()

    print("  → Grafik EDA tersimpan")

    # ── 4. TRAINING 5 MODEL ───────────────────────────────────────
    print("\n[4/7] Training 5 algoritma ML...")
    results = {}

    def ev(y_true, y_pred, y_prob=None):
        return (accuracy_score(y_true, y_pred),
                mean_absolute_error(y_true, y_pred),
                float(np.sqrt(mean_squared_error(y_true, y_pred))),
                r2_score(y_true, y_prob if y_prob is not None else y_pred),
                confusion_matrix(y_true, y_pred))

    def save_r(name, acc, mae, rmse, r2, cm, extra=None):
        d = {"Accuracy":round(float(acc)*100,2),"MAE":round(float(mae),4),
             "RMSE":round(float(rmse),4),"R2":round(float(r2),4),"CM":cm.tolist()}
        if extra: d.update(extra)
        results[name] = d
        print(f"    {name:<30} Accuracy={acc*100:.2f}%  MAE={mae:.4f}  RMSE={rmse:.4f}")

    def loss_fig(model, title, fname, c1, c2):
        fig, ax = plt.subplots(figsize=(8,4))
        ax.plot(model.loss_curve_, label="Training Loss", color=c1, lw=2)
        if hasattr(model,"validation_scores_") and model.validation_scores_:
            ax.plot([1-s for s in model.validation_scores_],
                    label="Validation Loss", color=c2, lw=2, linestyle="--")
        ax.set(xlabel="Epoch", ylabel="Loss", title=title); ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(IMG_DIR,fname), dpi=110, bbox_inches="tight"); plt.close()

    # Model 1: Logistic Regression
    print("  ► [1/5] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train_sc, y_train)
    yp = lr.predict(X_test_sc); yprob = lr.predict_proba(X_test_sc)[:,1]
    save_r("Logistic Regression", *ev(y_test, yp, yprob))
    joblib.dump(lr, os.path.join(MODEL_DIR,"logistic_regression.pkl"))

    # Model 2: ANN
    print("  ► [2/5] ANN (MLP, 2 hidden layer, Adam)...")
    ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                        solver="adam", max_iter=300, random_state=42,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)
    ann.fit(X_train_sc, y_train)
    yp = ann.predict(X_test_sc); yprob = ann.predict_proba(X_test_sc)[:,1]
    save_r("ANN (MLP)", *ev(y_test, yp, yprob))
    joblib.dump(ann, os.path.join(MODEL_DIR,"ann_model.pkl"))
    loss_fig(ann,"ANN (MLP) — Loss Curve","loss_curve_ann.png","#4e8fde","#f44336")

    # Model 3: RNN/LSTM (Deep MLP simulasi)
    print("  ► [3/5] RNN/LSTM (Deep Sequential MLP, 5 layer)...")
    rnn = MLPClassifier(hidden_layer_sizes=(128,64,64,32,16), activation="tanh",
                        solver="adam", learning_rate="adaptive", batch_size=64,
                        max_iter=400, random_state=42,
                        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
    rnn.fit(X_train_sc, y_train)
    yp = rnn.predict(X_test_sc); yprob = rnn.predict_proba(X_test_sc)[:,1]
    save_r("RNN/LSTM (Deep MLP)", *ev(y_test, yp, yprob))
    joblib.dump(rnn, os.path.join(MODEL_DIR,"rnn_model.pkl"))
    loss_fig(rnn,"RNN/LSTM (Deep MLP) — Loss Curve","loss_curve_rnn.png","#9c27b0","#ff9800")

    # Model 4: K-Means
    print("  ► [4/5] K-Means Clustering (k=2)...")
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    km.fit(X_train_sc)
    cl  = km.predict(X_test_sc)
    sil = silhouette_score(X_test_sc, cl)
    cmap_km = {cid: mode(y_test[cl==cid], keepdims=True).mode[0] for cid in np.unique(cl)}
    yp_km = np.array([cmap_km[c] for c in cl])
    acc, mae, rmse, r2, cm = ev(y_test, yp_km)
    save_r("K-Means Clustering", acc, mae, rmse, r2, cm, {"Silhouette":round(float(sil),4)})
    joblib.dump(km, os.path.join(MODEL_DIR,"kmeans_model.pkl"))

    pca2 = PCA(n_components=2, random_state=42)
    X2d  = pca2.fit_transform(X_test_sc)
    fig, axes = plt.subplots(1,2,figsize=(12,4))
    fig.suptitle("K-Means Clustering — Visualisasi 2D (PCA)", fontsize=13, fontweight="bold")
    sc1=axes[0].scatter(X2d[:,0],X2d[:,1],c=cl,cmap="coolwarm",alpha=0.4,s=8)
    axes[0].set_title("Hasil Cluster"); plt.colorbar(sc1,ax=axes[0])
    sc2=axes[1].scatter(X2d[:,0],X2d[:,1],c=y_test,cmap="coolwarm",alpha=0.4,s=8)
    axes[1].set_title("Label Aktual (0/1)"); plt.colorbar(sc2,ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,"kmeans_cluster.png"), dpi=110, bbox_inches="tight"); plt.close()
    joblib.dump(pca2, os.path.join(MODEL_DIR,"pca.pkl"))

    # Model 5: Backpropagation
    print("  ► [5/5] Backpropagation (SGD, sigmoid)...")
    bp = MLPClassifier(hidden_layer_sizes=(50,30), activation="logistic",
                       solver="sgd", learning_rate="constant",
                       learning_rate_init=0.01, momentum=0.9,
                       max_iter=500, random_state=42,
                       early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
    bp.fit(X_train_sc, y_train)
    yp = bp.predict(X_test_sc); yprob = bp.predict_proba(X_test_sc)[:,1]
    save_r("Backpropagation", *ev(y_test, yp, yprob))
    joblib.dump(bp, os.path.join(MODEL_DIR,"backpropagation.pkl"))
    loss_fig(bp,"Backpropagation (SGD) — Loss Curve","loss_curve_bp.png","#e91e63","#009688")

    # ── 5. CONFUSION MATRIX ───────────────────────────────────────
    print("\n[5/7] Confusion matrix...")
    fig, axes = plt.subplots(1,5,figsize=(22,4))
    fig.suptitle("Confusion Matrix — 5 Algoritma ML", fontsize=13, fontweight="bold")
    for ax, (name, r) in zip(axes, results.items()):
        sns.heatmap(np.array(r["CM"]), annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Tidak Lulus","Lulus"],
                    yticklabels=["Tidak Lulus","Lulus"], linewidths=0.5)
        ax.set_title(name, fontsize=8); ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,"confusion_matrix_all.png"), dpi=110, bbox_inches="tight"); plt.close()

    # ── 6. GRAFIK PERBANDINGAN ────────────────────────────────────
    print("\n[6/7] Grafik perbandingan model...")
    mnames = list(results.keys())
    colors = ["#4e8fde","#4caf50","#9c27b0","#ff9800","#e91e63"]
    fig, axes = plt.subplots(2,2,figsize=(14,9))
    fig.suptitle("Perbandingan Performa 5 Algoritma ML", fontsize=14, fontweight="bold")
    for ax, key, title in zip(
        [axes[0,0],axes[0,1],axes[1,0],axes[1,1]],
        ["Accuracy","MAE","RMSE","R2"],
        ["Accuracy (%)","MAE (↓ lebih baik)","RMSE (↓ lebih baik)","R² Score"]
    ):
        vals = [results[m][key] for m in mnames]
        bars = ax.bar(mnames, vals, color=colors, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xticklabels(mnames, rotation=15, ha="right", fontsize=8)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(),0)+0.002,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR,"perbandingan_model.png"), dpi=110, bbox_inches="tight"); plt.close()

    # ── 7. SIMPAN METADATA ────────────────────────────────────────
    print("\n[7/7] Menyimpan metadata...")
    with open(os.path.join(MODEL_DIR,"results.json"),"w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda m: results[m]["Accuracy"])
    meta = {"features":FEATURES,"target":TARGET,
            "n_train":int(len(X_train)),"n_test":int(len(X_test)),
            "best_model":best,"best_accuracy":results[best]["Accuracy"],
            "threshold":float(threshold)}
    with open(os.path.join(MODEL_DIR,"meta.json"),"w") as f:
        json.dump(meta, f, indent=2)

    # Simpan desc stats
    df[FEATURES+["Passed_Binary"]].describe().round(2)\
        .to_csv(os.path.join(MODEL_DIR,"desc_stats.csv"))

    print("\n" + "="*60)
    print("  RINGKASAN HASIL TRAINING")
    print("="*60)
    print(f"  {'Model':<28} {'Accuracy':>10} {'MAE':>8} {'RMSE':>8}")
    print("  " + "-"*57)
    for m, r in results.items():
        flag = " ← TERBAIK" if m == best else ""
        print(f"  {m:<28} {r['Accuracy']:>9.2f}% {r['MAE']:>8.4f} {r['RMSE']:>8.4f}{flag}")
    print("="*60)
    print(f"\n  ✓ TRAINING SELESAI — Model terbaik: {best} ({results[best]['Accuracy']}%)")
    print()


# ══════════════════════════════════════════════════════════════════
#  CEK MODEL — training otomatis jika belum ada
# ══════════════════════════════════════════════════════════════════
MODEL_FILES = [
    "scaler.pkl", "logistic_regression.pkl", "ann_model.pkl",
    "rnn_model.pkl", "kmeans_model.pkl", "backpropagation.pkl",
    "results.json", "meta.json", "prep_meta.json"
]

def models_exist():
    return all(os.path.exists(os.path.join(MODEL_DIR, f)) for f in MODEL_FILES)

def grafik_exist():
    grafik = ["dist_fitur.png","heatmap_korelasi.png","perbandingan_model.png",
              "confusion_matrix_all.png","kmeans_cluster.png"]
    return all(os.path.exists(os.path.join(IMG_DIR, g)) for g in grafik)

if not models_exist() or not grafik_exist():
    if not os.path.exists(DATA_PATH):
        print("=" * 60)
        print("  ERROR: Dataset tidak ditemukan!")
        print(f"  Letakkan file CSV di: {DATA_PATH}")
        print("=" * 60)
        sys.exit(1)
    run_training()
else:
    print("=" * 60)
    print("  Model sudah ada — skip training, langsung load model")
    print("=" * 60)


# ══════════════════════════════════════════════════════════════════
#  FLASK APPLICATION
# ══════════════════════════════════════════════════════════════════
from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__,
            template_folder=os.path.join(BASE_DIR, "app", "templates"),
            static_folder=os.path.join(BASE_DIR, "app", "static"))

# ── Load semua model ──────────────────────────────────────────────
print("\n[Flask] Memuat model ke memori...")
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

MODELS = {
    "logistic": ("Logistic Regression",   lr_model,  "#4e8fde"),
    "ann":      ("ANN (MLP)",             ann_model, "#4caf50"),
    "rnn":      ("RNN/LSTM (Deep MLP)",   rnn_model, "#9c27b0"),
    "kmeans":   ("K-Means Clustering",    km_model,  "#ff9800"),
    "backprop": ("Backpropagation",       bp_model,  "#e91e63"),
}

EDUCATION_MAP = {
    "High School":0, "Associate":1, "Bachelor":2, "Master":3, "PhD":4
}

print(f"[Flask] Semua model dimuat ✓  (Model terbaik: {meta['best_model']})")


# ── Helper functions ──────────────────────────────────────────────
def preprocess_input(study_hours, attendance, prev_grades, participation, parent_edu):
    participation_enc = 1 if participation == "Yes" else 0
    parent_edu_enc    = EDUCATION_MAP.get(parent_edu, 2)
    features = np.array([[float(study_hours), float(attendance),
                           float(prev_grades), float(participation_enc),
                           float(parent_edu_enc)]])
    return features, scaler.transform(features)


def predict_all_models(features_scaled):
    predictions = {}
    for key, (name, model, color) in MODELS.items():
        if key == "kmeans":
            cluster   = model.predict(features_scaled)[0]
            centroids = model.cluster_centers_
            pred = 1 if np.mean(centroids[cluster]) > np.mean(centroids[1-cluster]) else 0
            prob = 0.75 if pred == 1 else 0.25
        else:
            pred = int(model.predict(features_scaled)[0])
            prob = float(model.predict_proba(features_scaled)[0][1])

        predictions[key] = {
            "name":        name,
            "color":       color,
            "prediction":  pred,
            "label":       "LULUS" if pred == 1 else "TIDAK LULUS",
            "probability": round(prob * 100, 1),
            "confidence":  "Tinggi" if abs(prob - 0.5) > 0.3 else "Sedang"
        }
    return predictions


# ── ROUTES ────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html",
                           best_model=meta["best_model"],
                           best_accuracy=meta["best_accuracy"])


@app.route("/predict", methods=["POST"])
def predict():
    try:
        nama          = request.form.get("nama", "Mahasiswa")
        nim           = request.form.get("nim", "-")
        study_hours   = float(request.form.get("study_hours",  10))
        attendance    = float(request.form.get("attendance",   75))
        prev_grades   = float(request.form.get("prev_grades",  65))
        participation = request.form.get("participation", "No")
        parent_edu    = request.form.get("parent_edu",   "Bachelor")
        selected_model = request.form.get("selected_model", "backprop")

        study_hours = max(0, min(40,  study_hours))
        attendance  = max(0, min(100, attendance))
        prev_grades = max(0, min(100, prev_grades))

        features_raw, features_scaled = preprocess_input(
            study_hours, attendance, prev_grades, participation, parent_edu)

        all_predictions = predict_all_models(features_scaled)

        academic_score = (prev_grades * 0.50 +
                          attendance  * 0.35 +
                          study_hours * 1.50)

        votes    = [all_predictions[k]["prediction"]
                    for k in all_predictions if k != "kmeans"]
        majority = 1 if sum(votes) > len(votes) / 2 else 0

        input_data = {
            "nama": nama, "nim": nim,
            "study_hours": study_hours, "attendance": attendance,
            "prev_grades": prev_grades, "participation": participation,
            "parent_edu":  parent_edu,
            "academic_score": round(academic_score, 2),
            "threshold":      round(THRESHOLD, 2)
        }

        return render_template("result.html",
                               input_data=input_data,
                               all_predictions=all_predictions,
                               selected_model=selected_model,
                               majority_vote=majority,
                               majority_label="LULUS" if majority == 1 else "TIDAK LULUS")

    except Exception as e:
        return render_template("index.html", error=f"Error: {str(e)}",
                               best_model=meta["best_model"],
                               best_accuracy=meta["best_accuracy"])


@app.route("/compare")
def compare():
    ranked = sorted(model_results.items(),
                    key=lambda x: x[1]["Accuracy"], reverse=True)
    return render_template("compare.html",
                           results=model_results, ranked=ranked,
                           best_model=meta["best_model"], meta=meta)


@app.route("/eda")
def eda():
    return render_template("eda.html", meta=meta)


@app.route("/about")
def about():
    return render_template("about.html", meta=meta)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    try:
        data = request.get_json()
        _, features_scaled = preprocess_input(
            data.get("study_hours",  10), data.get("attendance",   75),
            data.get("prev_grades",  65), data.get("participation","No"),
            data.get("parent_edu", "Bachelor"))
        return jsonify({"status":"success",
                        "predictions": predict_all_models(features_scaled)})
    except Exception as e:
        return jsonify({"status":"error","message":str(e)}), 400


# ── ENTRY POINT ───────────────────────────────────────────────────
if __name__ == "__main__":
    PORT = 5000

    print("\n" + "=" * 60)
    print("  APLIKASI SIAP DIGUNAKAN!")
    print("=" * 60)
    print(f"  Buka browser dan akses: http://localhost:{PORT}")
    print(f"  Tekan Ctrl+C untuk menghentikan aplikasi")
    print("=" * 60 + "\n")

    # Buka browser otomatis setelah 1.5 detik
    def open_browser():
        import time
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{PORT}")

    threading.Thread(target=open_browser, daemon=True).start()

    app.run(debug=False, host="0.0.0.0", port=PORT)
