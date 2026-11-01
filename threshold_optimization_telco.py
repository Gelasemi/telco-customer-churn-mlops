
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = "preprocessing_results"
RESULT_DIR = "threshold_results"

os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 60)
print("TELCO CUSTOMER CHURN - THRESHOLD OPTIMIZATION")
print("=" * 60)

# =====================================================
# 2. CHARGEMENT DES DONNEES
# =====================================================

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze().astype(int)
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze().astype(int)

# =====================================================
# 3. SEPARATION TRAIN / VALIDATION
# =====================================================

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nDimensions :")
print("Train :", X_train_sub.shape)
print("Validation :", X_val.shape)
print("Test :", X_test.shape)

# =====================================================
# 4. ENTRAINEMENT DU MODELE
# =====================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

print("\nEntrainement Logistic Regression...")

model.fit(X_train_sub, y_train_sub)

# Probabilites de la classe Churn = 1
y_val_proba = model.predict_proba(X_val)[:, 1]

# =====================================================
# 5. COMPARAISON DES SEUILS
# =====================================================

thresholds = np.arange(0.20, 0.81, 0.05)

results = []

for threshold in thresholds:

    y_val_pred = (y_val_proba >= threshold).astype(int)

    results.append({
        "Threshold": round(threshold, 2),
        "Accuracy": accuracy_score(y_val, y_val_pred),
        "Precision": precision_score(
            y_val, y_val_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_val_pred, zero_division=0
        ),
        "F1": f1_score(
            y_val, y_val_pred, zero_division=0
        )
    })

results_df = pd.DataFrame(results)

results_df.to_csv(
    f"{RESULT_DIR}/threshold_comparison.csv",
    index=False
)

print("\nCOMPARAISON DES SEUILS")
print(results_df.to_string(index=False))

# =====================================================
# 6. SELECTION DU SEUIL
# =====================================================

# Meilleur compromis selon le F1-score
best_row = results_df.sort_values(
    by=["F1", "Recall", "Precision"],
    ascending=False
).iloc[0]

best_threshold = float(best_row["Threshold"])

print("\nSEUIL OPTIMAL SELON LE F1-SCORE")
print("Threshold :", best_threshold)
print("Precision :", round(best_row["Precision"], 4))
print("Recall :", round(best_row["Recall"], 4))
print("F1-score :", round(best_row["F1"], 4))

# =====================================================
# 7. VISUALISATION DES METRIQUES
# =====================================================

plt.figure(figsize=(10, 6))

plt.plot(
    results_df["Threshold"],
    results_df["Precision"],
    marker="o",
    label="Precision"
)

plt.plot(
    results_df["Threshold"],
    results_df["Recall"],
    marker="o",
    label="Recall"
)

plt.plot(
    results_df["Threshold"],
    results_df["F1"],
    marker="o",
    label="F1-score"
)

plt.axvline(
    best_threshold,
    linestyle="--",
    label=f"Best threshold = {best_threshold:.2f}"
)

plt.xlabel("Classification Threshold")
plt.ylabel("Score")
plt.title("Threshold Optimization - Telco Churn")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/threshold_optimization.png",
    dpi=150
)

plt.close()

# =====================================================
# 8. REENTRAINEMENT SUR TOUT LE TRAIN
# =====================================================

print("\nReentrainement sur tout le jeu train...")

final_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

final_model.fit(X_train, y_train)

# =====================================================
# 9. EVALUATION FINALE SUR LE TEST
# =====================================================

y_test_proba = final_model.predict_proba(X_test)[:, 1]

# Seuil optimise sur validation
y_test_pred = (y_test_proba >= best_threshold).astype(int)

print("\nEVALUATION FINALE SUR TEST")

print("\nClassification report :")

from sklearn.metrics import classification_report

print(classification_report(
    y_test,
    y_test_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

final_metrics = {
    "Threshold": best_threshold,
    "Accuracy": accuracy_score(y_test, y_test_pred),
    "Precision": precision_score(
        y_test, y_test_pred, zero_division=0
    ),
    "Recall": recall_score(
        y_test, y_test_pred, zero_division=0
    ),
    "F1": f1_score(
        y_test, y_test_pred, zero_division=0
    ),
    "ROC_AUC": roc_auc_score(y_test, y_test_proba),
    "Average_Precision": average_precision_score(
        y_test, y_test_proba
    )
}

print("\nMETRIQUES FINALES")

for key, value in final_metrics.items():
    print(f"{key}: {value:.4f}")

# =====================================================
# 10. MATRICE DE CONFUSION
# =====================================================

cm = confusion_matrix(y_test, y_test_pred)

print("\nMatrice de confusion :")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Churn", "Churn"]
)

disp.plot(cmap="Blues", values_format="d")
plt.title(f"Optimized Threshold = {best_threshold:.2f}")
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/confusion_matrix_optimized.png",
    dpi=150
)

plt.close()

# =====================================================
# 11. SAUVEGARDE DES RESULTATS
# =====================================================

pd.DataFrame([final_metrics]).to_csv(
    f"{RESULT_DIR}/final_threshold_metrics.csv",
    index=False
)

joblib.dump(
    final_model,
    f"{RESULT_DIR}/optimized_logistic_model.joblib"
)

with open(
    f"{RESULT_DIR}/selected_threshold.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(f"Selected threshold: {best_threshold:.2f}\n")
    file.write("Selection criterion: Best validation F1-score\n")

print("\n" + "=" * 60)
print("THRESHOLD OPTIMIZATION TERMINEE")
print("=" * 60)

print("Resultats disponibles dans :", RESULT_DIR)
