
import os
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# 1. CONFIGURATION
# =====================================================

DATA_DIR = "preprocessing_results"
MODEL_DIR = "threshold_results"
RESULT_DIR = "shap_results"

os.makedirs(RESULT_DIR, exist_ok=True)

print("=" * 60)
print("TELCO CUSTOMER CHURN - SHAP INTERPRETATION")
print("=" * 60)

# =====================================================
# 2. CHARGEMENT DES DONNEES ET DU MODELE
# =====================================================

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test = pd.read_csv(f"{DATA_DIR}/X_test.csv")

model = joblib.load(
    f"{MODEL_DIR}/optimized_logistic_model.joblib"
)

print("\nModele charge :", type(model).__name__)
print("Dimensions train :", X_train.shape)
print("Dimensions test :", X_test.shape)

# =====================================================
# 3. SELECTION DES DONNEES SHAP
# =====================================================

# Echantillon de reference pour expliquer le modele
X_background = X_train.sample(
    n=min(100, len(X_train)),
    random_state=42
)

# Echantillon a expliquer
X_explain = X_test.sample(
    n=min(500, len(X_test)),
    random_state=42
)

print("\nEchantillon de reference :", X_background.shape)
print("Echantillon a expliquer :", X_explain.shape)

# =====================================================
# 4. CREATION DE L'EXPLAINER
# =====================================================

print("\nCreation du SHAP LinearExplainer...")

explainer = shap.LinearExplainer(
    model,
    X_background
)

# Calcul des valeurs SHAP
shap_values = explainer(X_explain)

print("Calcul SHAP termine.")

# =====================================================
# 5. IMPORTANCE GLOBALE DES VARIABLES
# =====================================================

importance = np.abs(shap_values.values).mean(axis=0)

importance_df = pd.DataFrame({
    "Feature": X_explain.columns,
    "Mean_Absolute_SHAP": importance
})

importance_df = importance_df.sort_values(
    by="Mean_Absolute_SHAP",
    ascending=False
)

importance_df.to_csv(
    f"{RESULT_DIR}/shap_feature_importance.csv",
    index=False
)

print("\nTOP 15 VARIABLES LES PLUS IMPORTANTES")
print(importance_df.head(15).to_string(index=False))

# =====================================================
# 6. GRAPHIQUE GLOBAL : BEESWARM
# =====================================================

plt.figure()

shap.plots.beeswarm(
    shap_values,
    max_display=15,
    show=False
)

plt.title("SHAP Summary - Telco Customer Churn")
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/shap_summary.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 7. GRAPHIQUE GLOBAL : BAR PLOT
# =====================================================

plt.figure()

shap.plots.bar(
    shap_values,
    max_display=15,
    show=False
)

plt.title("Global Feature Importance - SHAP")
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/shap_bar.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 8. EXPLICATION D'UN CLIENT
# =====================================================

# On explique le premier client de l'echantillon
print("\nEXPLICATION INDIVIDUELLE DU PREMIER CLIENT")

print(X_explain.iloc[0])

plt.figure()

shap.plots.waterfall(
    shap_values[0],
    max_display=15,
    show=False
)

plt.title("SHAP Waterfall - Individual Customer")
plt.tight_layout()

plt.savefig(
    f"{RESULT_DIR}/shap_waterfall_customer.png",
    dpi=150,
    bbox_inches="tight"
)

plt.close()

# =====================================================
# 9. SAUVEGARDE DU RAPPORT
# =====================================================

with open(
    f"{RESULT_DIR}/shap_report.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("TELCO CUSTOMER CHURN - SHAP REPORT\n\n")

    file.write("Model: Logistic Regression\n")
    file.write("Explainer: LinearExplainer\n")
    file.write(f"Background observations: {len(X_background)}\n")
    file.write(f"Explained observations: {len(X_explain)}\n\n")

    file.write("TOP 15 FEATURES\n")
    file.write(importance_df.head(15).to_string(index=False))

print("\n" + "=" * 60)
print("SHAP INTERPRETATION TERMINEE")
print("=" * 60)

print("\nFichiers generes dans :", RESULT_DIR)
