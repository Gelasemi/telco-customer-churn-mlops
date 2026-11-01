
import os
import joblib
import pandas as pd

from typing import Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =====================================================
# 1. CONFIGURATION
# =====================================================

MODEL_PATH = "threshold_results/optimized_logistic_model.joblib"

PREPROCESSOR_PATH = (
    "preprocessing_results/preprocessor_telco.joblib"
)

THRESHOLD_PATH = "threshold_results/selected_threshold.txt"

# =====================================================
# 2. CHARGEMENT DES ARTEFACTS
# =====================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Modele introuvable : {MODEL_PATH}")

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocesseur introuvable : {PREPROCESSOR_PATH}"
    )

if not os.path.exists(THRESHOLD_PATH):
    raise FileNotFoundError(
        f"Seuil introuvable : {THRESHOLD_PATH}"
    )

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)
# Chargement du seuil optimal
with open("threshold_results/selected_threshold.txt", "r") as file:
    threshold_text = file.readline().strip()

threshold = float(threshold_text.split(":")[-1].strip())

print(f"Seuil de décision chargé : {threshold}")
print("Modele charge :", type(model).__name__)
print("Seuil de classification :", threshold)

# =====================================================
# 3. INITIALISATION FASTAPI
# =====================================================

app = FastAPI(
    title="Telco Customer Churn API",
    description=(
        "API de prediction du risque de resiliation "
        "des clients d'une entreprise telecom."
    ),
    version="1.0.0"
)

# =====================================================
# 4. STRUCTURE DES DONNEES CLIENT
# =====================================================

class CustomerData(BaseModel):

    tenure: int = Field(ge=0, le=100)
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]

    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]

    PhoneService: Literal["Yes", "No"]

    MultipleLines: Literal[
        "Yes",
        "No",
        "No phone service"
    ]

    InternetService: Literal[
        "DSL",
        "Fiber optic",
        "No"
    ]

    OnlineSecurity: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    OnlineBackup: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    DeviceProtection: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    TechSupport: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingTV: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    StreamingMovies: Literal[
        "Yes",
        "No",
        "No internet service"
    ]

    Contract: Literal[
        "Month-to-month",
        "One year",
        "Two year"
    ]

    PaperlessBilling: Literal["Yes", "No"]

    PaymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]

# =====================================================
# 5. ENDPOINT D'ACCUEIL
# =====================================================

@app.get("/")
def home():

    return {
        "message": "Telco Customer Churn API is running",
        "version": "1.0.0",
        "threshold": threshold,
        "model": "Logistic Regression"
    }

# =====================================================
# 6. ENDPOINT DE SANTE
# =====================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }

# =====================================================
# 7. ENDPOINT DE PREDICTION
# =====================================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:

        # Convertir les donnees en DataFrame
        customer_df = pd.DataFrame(
            [customer.model_dump()]
        )

        # Transformer les 19 variables en 46 variables
        customer_processed = preprocessor.transform(
            customer_df
        )

        # Calculer la probabilite de Churn
        probability = float(
            model.predict_proba(customer_processed)[0, 1]
        )

        # Appliquer le seuil optimise
        prediction = int(probability >= threshold)

        # Construire le resultat
        result = {
            "customer_churn_prediction": prediction,
            "churn_label": (
                "Churn" if prediction == 1 else "No Churn"
            ),
            "churn_probability": round(probability, 4),
            "churn_probability_percent": round(
                probability * 100, 2
            ),
            "classification_threshold": threshold,
            "risk_level": (
                "High" if prediction == 1 else "Below threshold"
            )
        }

        return result

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(error)}"
        )

# =====================================================
# 8. LANCEMENT
# =====================================================

# Lancement depuis le terminal :
# python -m uvicorn api:app --reload
