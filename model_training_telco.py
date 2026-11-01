
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = "preprocessing_results"
RESULT_DIR = "model_results"

os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 60)
print("TELCO CUSTOMER CHURN - MODEL TRAINING")
print("=" * 60)

# =====================================================
# 2. CHARGEMENT DES DONNEES
# =====================================================

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

y_train = pd.read_csv(f"{DATA_DIR}/y_train.csv").squeeze()
y_test = pd.read_csv(f"{DATA_DIR}/y_test.csv").squeeze()

y_train = y_train.astype(int)
y_test = y_test.astype(int)

print("\nDimensions des donnees :")
print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)

# =====================================================
# 3. CREATION DU JEU DE VALIDATION
# =====================================================

X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.20,
    random_state=42,
    stratify=y_train
)

print("\nTrain :", X_train_sub.shape)
print("Validation :", X_val.shape)

# =====================================================
# 4. DEFINITION DES MODELES
# =====================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
}

# =====================================================
# 5. ENTRAINEMENT ET EVALUATION
# =====================================================

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nEntrainement : {name}")

    model.fit(X_train_sub, y_train_sub)

    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "ROC_AUC": roc_auc_score(y_val, y_proba),
        "Average_Precision": average_precision_score(
            y_val, y_proba
        )
    }

    results.append(metrics)
    trained_models[name] = model

    print("Accuracy :", round(metrics["Accuracy"], 4))
    print("Precision:", round(metrics["Precision"], 4))
    print("Recall   :", round(metrics["Recall"], 4))
    print("F1       :", round(metrics["F1"], 4))
    print("ROC-AUC  :", round(metrics["ROC_AUC"], 4))
    print("Average Precision:", round(
        metrics["Average_Precision"], 4
    ))

# =====================================================
# 6. COMPARAISON DES MODELES
# =====================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Average_Precision",
    ascending=False
)

results_df.to_csv(
    f"{RESULT_DIR}/validation_comparison.csv",
    index=False
)

print("\n" + "=" * 60)
print("COMPARAISON DES MODELES")
print("=" * 60)

print(results_df.to_string(index=False))

# =====================================================
# 7. SELECTION DU MEILLEUR MODELE
# =====================================================

best_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_name]

print("\nMeilleur modele :", best_name)

# =====================================================
# 8. REENTRAINEMENT SUR TOUT LE TRAIN
# =====================================================

print("\nReentrainement sur toutes les donnees train...")

best_model.fit(X_train, y_train)

# =====================================================
# 9. EVALUATION FINALE SUR TEST
# =====================================================

y_test_pred = best_model.predict(X_test)
y_test_proba = best_model.predict_proba(X_test)[:, 1]

print("\n" + "=" * 60)
print("EVALUATION FINALE")
print("=" * 60)

print("\nClassification Report :")
print(classification_report(
    y_test,
    y_test_pred,
    target_names=["No Churn", "Churn"],
    zero_division=0
))

final_metrics = {
    "Model": best_name,
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

print("\nMetriques finales :")

for metric, value in final_metrics.items():
    print(f"{metric}: {value if isinstance(value, str) else round(value, 4)}")

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
plt.title(f"Confusion Matrix - {best_name}")
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/final_confusion_matrix.png",
    dpi=150
)

plt.close()

# =====================================================
# 11. SAUVEGARDE DU MODELE ET DES RESULTATS
# =====================================================

joblib.dump(
    best_model,
    f"{RESULT_DIR}/best_model.joblib"
)

pd.DataFrame([final_metrics]).to_csv(
    f"{RESULT_DIR}/final_test_metrics.csv",
    index=False
)

with open(
    f"{RESULT_DIR}/training_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("TELCO CUSTOMER CHURN - TRAINING REPORT\n\n")
    file.write(f"Best model: {best_name}\n\n")
    file.write(results_df.to_string(index=False))
    file.write("\n\nFINAL TEST METRICS\n")
    file.write(str(final_metrics))
    file.write("\n\nCONFUSION MATRIX\n")
    file.write(str(cm))

# =====================================================
# 12. FEATURE IMPORTANCE
# =====================================================

if best_name == "Random Forest":

    importance_df = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": best_model.feature_importances_
    }).sort_values(
        by="Importance",
        ascending=False
    )

    importance_df.to_csv(
        f"{RESULT_DIR}/feature_importance.csv",
        index=False
    )

    plt.figure(figsize=(10, 8))

    top_features = importance_df.head(15).sort_values(
        by="Importance"
    )

    plt.barh(
        top_features["Feature"],
        top_features["Importance"]
    )

    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()

    plt.savefig(
        f"{RESULT_DIR}/feature_importance.png",
        dpi=150
    )

    plt.close()

print("\n" + "=" * 60)
print("MODEL TRAINING TERMINE AVEC SUCCES")
print("=" * 60)

print("\nFichiers generes dans :", RESULT_DIR)
