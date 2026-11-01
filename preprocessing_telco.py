
# ============================================================
# TP : DATA PREPROCESSING
# Projet : TELCO CUSTOMER CHURN
# Python 3.11
# ============================================================

import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


# ============================================================
# 1. CONFIGURATION ET CHARGEMENT DES DONNEES
# ============================================================

print("=" * 70)
print("DATA PREPROCESSING - TELCO CUSTOMER CHURN")
print("=" * 70)

BASE_DIR = Path(__file__).resolve().parent

file_path = BASE_DIR / "WA_Fn-UseC_-Telco-Customer-Churn (1).csv"

output_dir = BASE_DIR / "preprocessing_results"
output_dir.mkdir(exist_ok=True)

if not file_path.exists():
    raise FileNotFoundError(f"Dataset introuvable : {file_path}")

df = pd.read_csv(file_path)

print("\n1. DATASET INITIAL")
print("Dimensions :", df.shape)


# ============================================================
# 2. VERIFICATION DE LA STRUCTURE
# ============================================================

print("\n2. VERIFICATION DE LA STRUCTURE")

required_columns = [
    "customerID",
    "Churn",
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Colonnes absentes du dataset : {missing_columns}"
    )

# Supprimer les espaces inutiles des colonnes texte
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = df[col].str.strip()

# Vérification des identifiants
if df["customerID"].isnull().any():
    raise ValueError("Des identifiants client sont manquants.")

if df["customerID"].duplicated().any():
    raise ValueError("Des identifiants client sont dupliqués.")


# ============================================================
# 3. NETTOYAGE DES DONNEES
# ============================================================

print("\n3. NETTOYAGE DES DONNEES")

# Conversion de TotalCharges en numérique
df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

missing_before = df["TotalCharges"].isna().sum()

print("Valeurs manquantes TotalCharges :", missing_before)

# Identifier les valeurs manquantes avec tenure = 0
zero_tenure_mask = (
    df["TotalCharges"].isna()
    & (df["tenure"] == 0)
)

print(
    "TotalCharges manquant avec tenure = 0 :",
    zero_tenure_mask.sum()
)

# Règle métier : tenure = 0 implique aucun frais cumulé
df.loc[zero_tenure_mask, "TotalCharges"] = 0

print(
    "Valeurs manquantes restantes :",
    df["TotalCharges"].isna().sum()
)

# Les éventuelles autres valeurs manquantes seront
# traitées par SimpleImputer sur les données train.


# ============================================================
# 4. IDENTIFICATION DE LA VARIABLE CIBLE
# ============================================================

print("\n4. IDENTIFICATION DE LA CIBLE")

target = "Churn"

if df[target].isna().any():
    raise ValueError("La variable Churn contient des valeurs manquantes.")

if not df[target].isin(["Yes", "No"]).all():
    raise ValueError(
        "La variable Churn contient des modalités inattendues."
    )

# Encodage binaire de la cible
y = df[target].map({
    "No": 0,
    "Yes": 1
}).astype("int64")

y.name = "Churn"

print("Churn : No = 0 ; Yes = 1")

print("\nRépartition de la cible :")
print(y.value_counts())

print("\nProportions :")
print(y.value_counts(normalize=True).round(3))


# ============================================================
# 5. CONSTRUCTION DES VARIABLES EXPLICATIVES
# ============================================================

print("\n5. CONSTRUCTION DE X")

# Suppression de l'identifiant et de la cible
X = df.drop(columns=["customerID", "Churn"])

print("Nombre de variables explicatives :", X.shape[1])

# Variables numériques
numeric_features = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

# Vérifier leur présence
for col in numeric_features:
    if col not in X.columns:
        raise ValueError(f"Variable numérique absente : {col}")

# Variables catégorielles
categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

print("\nVariables quantitatives :")
print(numeric_features)

print("\nVariables catégorielles :")
print(categorical_features)


# ============================================================
# 6. SEPARATION TRAIN / TEST
# ============================================================

print("\n6. SEPARATION TRAIN / TEST")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("X_train :", X_train.shape)
print("X_test  :", X_test.shape)
print("y_train :", y_train.shape)
print("y_test  :", y_test.shape)

print("\nRépartition de Churn dans le train :")
print(y_train.value_counts(normalize=True).round(3))

print("\nRépartition de Churn dans le test :")
print(y_test.value_counts(normalize=True).round(3))


# ============================================================
# 7. PIPELINE NUMERIQUE
# ============================================================

print("\n7. PREPARATION DES VARIABLES NUMERIQUES")

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ============================================================
# 8. PIPELINE CATEGORIEL
# ============================================================

print("\n8. PREPARATION DES VARIABLES CATEGORIELLES")

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# ============================================================
# 9. ASSEMBLAGE DU PREPROCESSING
# ============================================================

print("\n9. ASSEMBLAGE DU PIPELINE")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)


# ============================================================
# 10. APPRENTISSAGE ET TRANSFORMATION
# ============================================================

print("\n10. TRANSFORMATION DES DONNEES")

# Apprentissage uniquement sur le train
X_train_processed = preprocessor.fit_transform(X_train)

# Transformation du test sans réapprentissage
X_test_processed = preprocessor.transform(X_test)

print(
    "Dimensions X_train après preprocessing :",
    X_train_processed.shape
)

print(
    "Dimensions X_test après preprocessing :",
    X_test_processed.shape
)


# ============================================================
# 11. NOMS DES VARIABLES TRANSFORMEES
# ============================================================

print("\n11. VARIABLES APRES ENCODAGE")

feature_names = preprocessor.get_feature_names_out()

print("Nombre de variables finales :", len(feature_names))

print("\nExemples de variables transformées :")

for feature in feature_names[:20]:
    print("-", feature)


# ============================================================
# 12. CREATION DES DATAFRAMES FINAUX
# ============================================================

print("\n12. CREATION DES DATASETS FINAUX")

X_train_df = pd.DataFrame(
    X_train_processed,
    columns=feature_names,
    index=X_train.index
)

X_test_df = pd.DataFrame(
    X_test_processed,
    columns=feature_names,
    index=X_test.index
)

# Ajouter la cible en conservant l'alignement des index
train_final = X_train_df.copy()
train_final["Churn"] = y_train

test_final = X_test_df.copy()
test_final["Churn"] = y_test

print("Train final :", train_final.shape)
print("Test final  :", test_final.shape)


# ============================================================
# 13. EXPORT DES RESULTATS
# ============================================================

print("\n13. EXPORT DES RESULTATS")

# Export des variables explicatives
X_train_df.to_csv(
    output_dir / "X_train.csv",
    index=False
)

X_test_df.to_csv(
    output_dir / "X_test.csv",
    index=False
)

# Export des cibles
y_train.to_csv(
    output_dir / "y_train.csv",
    index=False
)

y_test.to_csv(
    output_dir / "y_test.csv",
    index=False
)

# Export des datasets complets
train_final.to_csv(
    output_dir / "train_final.csv",
    index=False
)

test_final.to_csv(
    output_dir / "test_final.csv",
    index=False
)

# Sauvegarde du préprocesseur
joblib.dump(
    preprocessor,
    output_dir / "preprocessor_telco.joblib"
)

# Sauvegarde des noms des variables
pd.DataFrame({
    "feature": feature_names
}).to_csv(
    output_dir / "feature_names.csv",
    index=False
)


# ============================================================
# 14. BILAN FINAL
# ============================================================

print("\n" + "=" * 70)
print("BILAN FINAL DU DATA PREPROCESSING")
print("=" * 70)

print("Dataset initial :", df.shape)
print("Train :", X_train_processed.shape)
print("Test :", X_test_processed.shape)

print("Nombre de variables finales :", len(feature_names))

print("\nNombre de clients dans le train :", len(y_train))
print("Nombre de clients dans le test :", len(y_test))

print("\nFichiers enregistrés dans :", output_dir.resolve())

print("\nFichiers générés :")

for filename in sorted(output_dir.iterdir()):
    if filename.is_file():
        print("-", filename.name)

print("\nDATA PREPROCESSING TERMINE AVEC SUCCES.")
