
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. CHARGEMENT DES DONNEES
# ============================================================

print("=" * 65)
print("DATA QUALITY ASSESSMENT - TELCO CUSTOMER CHURN")
print("=" * 65)

file_path = Path(
    r"C:\Users\hp\Desktop\telecom\WA_Fn-UseC_-Telco-Customer-Churn (1).csv"
)

if not file_path.exists():
    raise FileNotFoundError(f"Fichier introuvable : {file_path}")

df = pd.read_csv(file_path)

print("\n1. STRUCTURE DU DATASET")
print("Nombre de lignes :", df.shape[0])
print("Nombre de colonnes :", df.shape[1])

# Conversion de TotalCharges
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"].astype("string").str.strip(),
    errors="coerce"
)

# ============================================================
# 2. VALEURS MANQUANTES
# ============================================================

print("\n2. VALEURS MANQUANTES")

missing = pd.DataFrame({
    "Nombre": df.isnull().sum(),
    "Pourcentage": (df.isnull().mean() * 100).round(2)
})

print(missing[missing["Nombre"] > 0])

# ============================================================
# 3. DOUBLONS
# ============================================================

print("\n3. DOUBLONS")

print("Lignes dupliquées :", df.duplicated().sum())

print(
    "Identifiants clients dupliqués :",
    df["customerID"].duplicated().sum()
)

# ============================================================
# 4. INCOHERENCES
# ============================================================

print("\n4. CONTROLE DES INCOHERENCES")

checks = {
    "Tenure négatif": (df["tenure"] < 0).sum(),
    "MonthlyCharges négatif": (df["MonthlyCharges"] < 0).sum(),
    "TotalCharges négatif": (df["TotalCharges"] < 0).sum(),
    "SeniorCitizen invalide": (~df["SeniorCitizen"].isin([0, 1])).sum(),
    "Churn invalide": (~df["Churn"].isin(["Yes", "No"])).sum()
}

for name, result in checks.items():
    print(f"{name} : {result}")

# ============================================================
# 5. ANALYSE DES VALEURS ABERRANTES PAR IQR
# ============================================================

print("\n5. DETECTION DES VALEURS ABERRANTES")

numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

outlier_results = []

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    count = (
        (df[col] < lower) |
        (df[col] > upper)
    ).sum()

    outlier_results.append({
        "Variable": col,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Borne basse": lower,
        "Borne haute": upper,
        "Nombre aberrant": count
    })

outlier_report = pd.DataFrame(outlier_results)

print(outlier_report.to_string(index=False))

# ============================================================
# 6. VISUALISATION DES VALEURS ABERRANTES
# ============================================================

print("\n6. VISUALISATION DES BOXPLOTS")

df[numeric_columns].plot(
    kind="box",
    subplots=True,
    layout=(1, 3),
    figsize=(14, 5),
    sharey=False
)

plt.suptitle("Détection des valeurs aberrantes - Telco")
plt.tight_layout()
plt.show()

# ============================================================
# 7. DISTRIBUTION DE CHURN
# ============================================================

print("\n7. DISTRIBUTION DE LA VARIABLE CIBLE")

print(df["Churn"].value_counts())

print("\nPourcentage :")
print(
    (df["Churn"].value_counts(normalize=True) * 100).round(2)
)

df["Churn"].value_counts().plot(
    kind="bar",
    rot=0,
    figsize=(7, 5)
)

plt.title("Distribution de Churn")
plt.xlabel("Churn")
plt.ylabel("Nombre de clients")
plt.tight_layout()
plt.show()

# ============================================================
# 8. NETTOYAGE PRUDENT
# ============================================================

print("\n8. PREPARATION DU DATASET")

df_clean = df.copy()

# Suppression des doublons exacts
df_clean = df_clean.drop_duplicates()

# Suppression des espaces inutiles dans les colonnes texte
for col in df_clean.select_dtypes(include=["object", "str"]).columns:
    df_clean[col] = df_clean[col].str.strip()

# Conservation des lignes avec identifiant et cible renseignés
df_clean = df_clean.dropna(
    subset=["customerID", "Churn"]
)

print("Dimensions initiales :", df.shape)
print("Dimensions nettoyées :", df_clean.shape)

# ============================================================
# 9. EXPORT DES RAPPORTS
# ============================================================

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

missing.to_csv(output_dir / "missing_values_report.csv")

outlier_report.to_csv(
    output_dir / "outlier_report.csv",
    index=False
)

df_clean.to_csv(
    output_dir / "telco_cleaned.csv",
    index=False
)

# ============================================================
# 10. BILAN FINAL
# ============================================================

print("\n" + "=" * 65)
print("BILAN FINAL")
print("=" * 65)

print("Dataset initial :", df.shape)
print("Dataset nettoyé :", df_clean.shape)

print("\nFichiers générés dans le dossier results :")
print("- missing_values_report.csv")
print("- outlier_report.csv")
print("- telco_cleaned.csv")

print("\nDATA QUALITY ASSESSMENT TERMINE.")
