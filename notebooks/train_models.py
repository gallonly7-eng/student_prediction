"""
=========================================================
  PROYEK UTS: Prediksi Kelulusan Mahasiswa
  File   : notebooks/train_models.py
  Fungsi : Preprocessing + EDA + Training 5 Algoritma ML
=========================================================
CATATAN DATASET:
  Dataset ini adalah dataset sintetis (buatan) dari Kaggle.
  Kolom "Passed" pada dataset asli tidak memiliki korelasi
  dengan fitur akademik (acak). Oleh karena itu, target
  direkonstruksi menggunakan formula skor akademik tertimbang
  yang merepresentasikan logika kelulusan nyata:
    score = (Nilai × 0.5) + (Kehadiran × 0.35) + (Jam Belajar × 1.5)
    Lulus = score >= median(score)

ALGORITMA:
  1. Logistic Regression  (analogi Linear Reg untuk klasifikasi)
  2. ANN  (MLPClassifier 2 hidden layer, Adam optimizer)
  3. RNN/LSTM simulasi (Deep MLP 5 layer, tanh, sequential)
  4. K-Means Clustering (unsupervised, k=2)
  5. Backpropagation (MLPClassifier SGD, sigmoid activation)
"""

# ─── IMPORT ───────────────────────────────────────────────────────────────────
import os, warnings, json
import numpy  as np
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

warnings.filterwarnings("ignore")

# ─── PATH ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_performance_prediction.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR   = os.path.join(BASE_DIR, "app", "static", "img")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(IMG_DIR,   exist_ok=True)

print("=" * 65)
print("  PREDIKSI KELULUSAN MAHASISWA — TRAINING PIPELINE")
print("=" * 65)

# ══════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════
print("\n[1/7] LOAD DATA ...")
df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["Student ID"], errors="ignore")
print(f"  → Dataset: {df.shape[0]:,} baris × {df.shape[1]} kolom")
print(f"  → Kolom  : {list(df.columns)}")

# ══════════════════════════════════════════════════════════════════
# 2. PREPROCESSING
# ══════════════════════════════════════════════════════════════════
print("\n[2/7] PREPROCESSING ...")
print("  Missing values (sebelum):")
for col in df.columns:
    mv = int(df[col].isnull().sum())
    if mv > 0:
        print(f"    {col:<45} {mv:>5} ({mv/len(df)*100:.1f}%)")

# Konversi numerik & isi missing
num_cols = ["Study Hours per Week", "Attendance Rate", "Previous Grades"]
cat_cols = ["Participation in Extracurricular Activities", "Parent Education Level"]

df[num_cols] = df[num_cols].apply(pd.to_numeric, errors="coerce")
for c in num_cols:
    df[c] = df[c].fillna(df[c].median())
for c in cat_cols:
    df[c] = df[c].fillna(df[c].mode()[0])

# Clip outlier (IQR 1.5)
for c in num_cols:
    Q1, Q3 = df[c].quantile(0.25), df[c].quantile(0.75)
    df[c]  = df[c].clip(Q1 - 1.5*(Q3-Q1), Q3 + 1.5*(Q3-Q1))

print(f"  → Data setelah cleaning: {len(df):,} baris")

# ── Encoding kategorikal ──────────────────────────────────────────
df["Participation_Enc"] = df["Participation in Extracurricular Activities"].map({"Yes":1,"No":0}).fillna(0).astype(int)
edu_map = {"High School":0, "Associate":1, "Bachelor":2, "Master":3, "PhD":4}
df["ParentEdu_Enc"] = df["Parent Education Level"].map(edu_map).fillna(2).astype(int)

# ── Rekonstruksi target yang meaningful ──────────────────────────
# Skor akademik tertimbang → threshold median → Lulus/Tidak
df["AcademicScore"] = (df["Previous Grades"] * 0.50 +
                        df["Attendance Rate"]  * 0.35 +
                        df["Study Hours per Week"] * 1.50)
threshold = df["AcademicScore"].median()
df["Passed_Binary"] = (df["AcademicScore"] >= threshold).astype(int)

print(f"  → Target rekonstruksi dari skor akademik (threshold={threshold:.2f})")
print(f"  → Distribusi: Lulus={df['Passed_Binary'].sum():,} | Tidak Lulus={(df['Passed_Binary']==0).sum():,}")
print("  → Encoding kategorikal selesai")

# ── Fitur & Target ────────────────────────────────────────────────
FEATURES = ["Study Hours per Week", "Attendance Rate",
            "Previous Grades", "Participation_Enc", "ParentEdu_Enc"]
TARGET   = "Passed_Binary"

X = df[FEATURES].values.astype(float)
y = df[TARGET].values.astype(int)

# Split 80:20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"  → Train: {len(X_train):,} | Test: {len(X_test):,} (split 80:20)")

# Standardisasi
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# Simpan threshold untuk Flask app
meta_prep = {"threshold": float(threshold), "features": FEATURES}
with open(os.path.join(MODEL_DIR, "prep_meta.json"), "w") as f:
    json.dump(meta_prep, f)
print("  → Scaler & metadata preprocessing disimpan")

# ══════════════════════════════════════════════════════════════════
# 3. EDA
# ══════════════════════════════════════════════════════════════════
print("\n[3/7] EDA — Exploratory Data Analysis ...")

# Statistik deskriptif
desc = df[FEATURES + ["Passed_Binary"]].describe().round(2)
desc.to_csv(os.path.join(MODEL_DIR, "desc_stats.csv"))

# 3a. Distribusi fitur numerik
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Distribusi Fitur Numerik Utama", fontsize=14, fontweight="bold")
for ax, col, color in zip(axes,
    ["Study Hours per Week","Attendance Rate","Previous Grades"],
    ["#4e8fde","#4caf50","#ff9800"]):
    sns.histplot(df[col], kde=True, ax=ax, color=color, bins=40)
    ax.set_title(col, fontsize=10); ax.set_xlabel("")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "dist_fitur.png"), dpi=110, bbox_inches="tight"); plt.close()

# 3b. Heatmap korelasi
corr = df[FEATURES + ["Passed_Binary"]].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
ax.set_title("Heatmap Korelasi Antar Fitur", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "heatmap_korelasi.png"), dpi=110, bbox_inches="tight"); plt.close()

# 3c. Distribusi target
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Analisis Distribusi Kelulusan", fontsize=13, fontweight="bold")
vc = df["Passed_Binary"].value_counts().sort_index()
axes[0].pie([vc.get(0,0), vc.get(1,0)],
            labels=["Tidak Lulus","Lulus"],
            autopct="%1.1f%%", colors=["#f44336","#4caf50"], startangle=90)
axes[0].set_title("Proporsi Kelulusan")
sns.boxplot(x="Passed_Binary", y="Previous Grades", data=df,
            palette={"0":"#f44336","1":"#4caf50"}, ax=axes[1])
axes[1].set_title("Previous Grades vs Kelulusan")
# axes[1].set_xticklabels(["Tidak Lulus","Lulus"])
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "target_dist.png"), dpi=110, bbox_inches="tight"); plt.close()

# 3d. Feature importance
fi = corr["Passed_Binary"].drop("Passed_Binary").abs().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(7, 4))
fi.plot(kind="barh", ax=ax, color="#4e8fde")
ax.set_title("Korelasi Fitur dengan Kelulusan (|nilai|)", fontsize=12, fontweight="bold")
ax.set_xlabel("Nilai Absolut Korelasi")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=110, bbox_inches="tight"); plt.close()
print("  → Semua grafik EDA tersimpan")

# ══════════════════════════════════════════════════════════════════
# 4. TRAINING 5 ALGORITMA
# ══════════════════════════════════════════════════════════════════
print("\n[4/7] TRAINING 5 ALGORITMA ...")
results = {}

def metrics(y_true, y_pred, y_prob=None):
    acc  = accuracy_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_prob if y_prob is not None else y_pred)
    cm   = confusion_matrix(y_true, y_pred)
    return acc, mae, rmse, r2, cm

def save_result(name, acc, mae, rmse, r2, cm):
    results[name] = {
        "Accuracy": round(float(acc)*100, 2),
        "MAE":  round(float(mae), 4),
        "RMSE": round(float(rmse), 4),
        "R2":   round(float(r2), 4),
        "CM":   cm.tolist()
    }
    print(f"    Accuracy={acc*100:.2f}% | MAE={mae:.4f} | RMSE={rmse:.4f} | R²={r2:.4f}")

def save_loss_curve(model, name, filename, color1, color2):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(model.loss_curve_, label="Training Loss", color=color1, lw=2)
    if hasattr(model, "validation_scores_") and model.validation_scores_:
        ax.plot([1-s for s in model.validation_scores_],
                label="Validation Loss", color=color2, lw=2, linestyle="--")
    ax.set(xlabel="Epoch", ylabel="Loss", title=f"{name} — Loss Curve")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, filename), dpi=110, bbox_inches="tight"); plt.close()

# ── MODEL 1: LOGISTIC REGRESSION ──────────────────────────────────
print("\n  ► Model 1: Logistic Regression")
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train_sc, y_train)
yp = lr.predict(X_test_sc);  yprob = lr.predict_proba(X_test_sc)[:,1]
save_result("Logistic Regression", *metrics(y_test, yp, yprob))
joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.pkl"))

# ── MODEL 2: ANN ──────────────────────────────────────────────────
print("\n  ► Model 2: ANN (MLPClassifier, 2 hidden layer, Adam)")
ann = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                    solver="adam", max_iter=300, random_state=42,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=15)
ann.fit(X_train_sc, y_train)
yp = ann.predict(X_test_sc);  yprob = ann.predict_proba(X_test_sc)[:,1]
save_result("ANN (MLP)", *metrics(y_test, yp, yprob))
joblib.dump(ann, os.path.join(MODEL_DIR, "ann_model.pkl"))
save_loss_curve(ann, "ANN (MLP)", "loss_curve_ann.png", "#4e8fde", "#f44336")

# ── MODEL 3: RNN/LSTM (Deep MLP simulasi) ─────────────────────────
print("\n  ► Model 3: RNN/LSTM (Deep Sequential MLP, 5 layer, tanh)")
rnn = MLPClassifier(hidden_layer_sizes=(128,64,64,32,16), activation="tanh",
                    solver="adam", learning_rate="adaptive", batch_size=64,
                    max_iter=400, random_state=42,
                    early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
rnn.fit(X_train_sc, y_train)
yp = rnn.predict(X_test_sc);  yprob = rnn.predict_proba(X_test_sc)[:,1]
save_result("RNN/LSTM (Deep MLP)", *metrics(y_test, yp, yprob))
joblib.dump(rnn, os.path.join(MODEL_DIR, "rnn_model.pkl"))
save_loss_curve(rnn, "RNN/LSTM (Deep MLP)", "loss_curve_rnn.png", "#9c27b0", "#ff9800")

# ── MODEL 4: K-MEANS ──────────────────────────────────────────────
print("\n  ► Model 4: K-Means Clustering (k=2)")
km = KMeans(n_clusters=2, random_state=42, n_init=10)
km.fit(X_train_sc)
cl = km.predict(X_test_sc)
sil = silhouette_score(X_test_sc, cl)
cmap_km = {}
for cid in np.unique(cl):
    cmap_km[cid] = mode(y_test[cl==cid], keepdims=True).mode[0]
yp_km = np.array([cmap_km[c] for c in cl])
acc, mae, rmse, r2, cm = metrics(y_test, yp_km)
results["K-Means Clustering"] = {
    "Accuracy": round(float(acc)*100,2), "MAE":round(float(mae),4),
    "RMSE":round(float(rmse),4), "R2":round(float(r2),4),
    "Silhouette":round(float(sil),4), "CM":cm.tolist()
}
print(f"    Accuracy={acc*100:.2f}% | Silhouette={sil:.4f} | MAE={mae:.4f} | RMSE={rmse:.4f}")
joblib.dump(km, os.path.join(MODEL_DIR, "kmeans_model.pkl"))

pca2 = PCA(n_components=2, random_state=42)
X2d  = pca2.fit_transform(X_test_sc)
fig, axes = plt.subplots(1,2,figsize=(12,4))
fig.suptitle("K-Means Clustering — Visualisasi 2D (PCA)", fontsize=13, fontweight="bold")
sc1 = axes[0].scatter(X2d[:,0],X2d[:,1],c=cl,cmap="coolwarm",alpha=0.4,s=8)
axes[0].set_title("Hasil Cluster K-Means"); plt.colorbar(sc1,ax=axes[0])
sc2 = axes[1].scatter(X2d[:,0],X2d[:,1],c=y_test,cmap="coolwarm",alpha=0.4,s=8)
axes[1].set_title("Label Aktual (0=Tidak Lulus / 1=Lulus)"); plt.colorbar(sc2,ax=axes[1])
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,"kmeans_cluster.png"), dpi=110, bbox_inches="tight"); plt.close()
joblib.dump(pca2, os.path.join(MODEL_DIR, "pca.pkl"))

# ── MODEL 5: BACKPROPAGATION (SGD) ────────────────────────────────
print("\n  ► Model 5: Backpropagation (SGD optimizer, sigmoid)")
bp = MLPClassifier(hidden_layer_sizes=(50,30), activation="logistic",
                   solver="sgd", learning_rate="constant",
                   learning_rate_init=0.01, momentum=0.9,
                   max_iter=500, random_state=42,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
bp.fit(X_train_sc, y_train)
yp = bp.predict(X_test_sc);  yprob = bp.predict_proba(X_test_sc)[:,1]
save_result("Backpropagation", *metrics(y_test, yp, yprob))
joblib.dump(bp, os.path.join(MODEL_DIR, "backpropagation.pkl"))
save_loss_curve(bp, "Backpropagation (SGD)", "loss_curve_bp.png", "#e91e63", "#009688")

# ══════════════════════════════════════════════════════════════════
# 5. CONFUSION MATRIX SEMUA MODEL
# ══════════════════════════════════════════════════════════════════
print("\n[5/7] CONFUSION MATRIX ...")
fig, axes = plt.subplots(1, 5, figsize=(22, 4))
fig.suptitle("Confusion Matrix — 5 Algoritma ML", fontsize=13, fontweight="bold")
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(np.array(r["CM"]), annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Tidak Lulus","Lulus"],
                yticklabels=["Tidak Lulus","Lulus"], linewidths=0.5)
    ax.set_title(name, fontsize=8); ax.set_xlabel("Prediksi"); ax.set_ylabel("Aktual")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,"confusion_matrix_all.png"), dpi=110, bbox_inches="tight"); plt.close()
print("  → Confusion matrix disimpan")

# ══════════════════════════════════════════════════════════════════
# 6. GRAFIK PERBANDINGAN
# ══════════════════════════════════════════════════════════════════
print("\n[6/7] GRAFIK PERBANDINGAN MODEL ...")
mnames = list(results.keys())
colors = ["#4e8fde","#4caf50","#9c27b0","#ff9800","#e91e63"]

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("Perbandingan Performa 5 Algoritma Machine Learning", fontsize=14, fontweight="bold")
for ax, key, title in zip(
    [axes[0,0], axes[0,1], axes[1,0], axes[1,1]],
    ["Accuracy","MAE","RMSE","R2"],
    ["Accuracy (%)","MAE (↓ lebih baik)","RMSE (↓ lebih baik)","R² Score"]
):
    vals = [results[m][key] for m in mnames]
    bars = ax.bar(mnames, vals, color=colors, edgecolor="white")
    ax.set_title(title, fontweight="bold")
    ax.set_xticklabels(mnames, rotation=15, ha="right", fontsize=8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, max(bar.get_height(), 0)+0.002,
                f"{val:.2f}", ha="center", va="bottom", fontsize=7, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR,"perbandingan_model.png"), dpi=110, bbox_inches="tight"); plt.close()
print("  → Grafik perbandingan disimpan")

# ══════════════════════════════════════════════════════════════════
# 7. SIMPAN SEMUA HASIL
# ══════════════════════════════════════════════════════════════════
print("\n[7/7] MENYIMPAN METADATA ...")
with open(os.path.join(MODEL_DIR,"results.json"),"w") as f:
    json.dump(results, f, indent=2)

best = max(results, key=lambda m: results[m]["Accuracy"])
meta = {
    "features":      FEATURES,
    "target":        TARGET,
    "n_train":       int(len(X_train)),
    "n_test":        int(len(X_test)),
    "best_model":    best,
    "best_accuracy": results[best]["Accuracy"],
    "threshold":     float(threshold)
}
with open(os.path.join(MODEL_DIR,"meta.json"),"w") as f:
    json.dump(meta, f, indent=2)

# ── RINGKASAN ─────────────────────────────────────────────────────
print("\n" + "="*65)
print("  RINGKASAN EVALUASI MODEL")
print("="*65)
print(f"  {'Model':<30} {'Accuracy':>10} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
print("  " + "-"*62)
for m, r in results.items():
    flag = " ← TERBAIK" if m == best else ""
    print(f"  {m:<30} {r['Accuracy']:>9.2f}% {r['MAE']:>8.4f} {r['RMSE']:>8.4f} {r['R2']:>8.4f}{flag}")
print("="*65)
print(f"\n  Model Terbaik : {best}")
print(f"  Accuracy      : {results[best]['Accuracy']:.2f}%")
print("\n  ✓ TRAINING SELESAI — semua file model & grafik tersimpan")
