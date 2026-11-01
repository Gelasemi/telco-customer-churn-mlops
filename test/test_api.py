
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

# Données de test : profil client
CUSTOMER_DATA = {
    "tenure": 12,
    "MonthlyCharges": 75.5,
    "TotalCharges": 900.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
}


def test_health_endpoint():
    """Vérifie que l'API est accessible."""
    response = client.get("/health")

    assert response.status_code == 200


def test_root_endpoint():
    """Vérifie la page d'accueil."""
    response = client.get("/")

    assert response.status_code == 200


def test_predict_endpoint():
    """Vérifie qu'une prédiction est correctement retournée."""
    response = client.post("/predict", json=CUSTOMER_DATA)

    assert response.status_code == 200

    result = response.json()

    assert "customer_churn_prediction" in result
    assert "churn_label" in result
    assert "churn_probability" in result
    assert "churn_probability_percent" in result
    assert "classification_threshold" in result
    assert "risk_level" in result

    assert result["customer_churn_prediction"] in [0, 1]
    assert result["churn_label"] in ["Churn", "No Churn"]

    probability = result["churn_probability"]
    percentage = result["churn_probability_percent"]
    threshold = result["classification_threshold"]

    assert 0 <= probability <= 1
    assert 0 <= percentage <= 100
    assert abs(percentage - probability * 100) < 0.01
    assert 0 <= threshold <= 1

    expected_prediction = int(probability >= threshold)
    assert result["customer_churn_prediction"] == expected_prediction


def test_predict_invalid_data():
    """Vérifie que l'API refuse une requête incomplète."""
    invalid_data = {
        "tenure": 12,
        "MonthlyCharges": 75.5
    }

    response = client.post("/predict", json=invalid_data)

    assert response.status_code == 422
