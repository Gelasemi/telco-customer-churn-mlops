# Telco Customer Churn Prediction – Machine Learning & API

**Projet de prédiction du désabonnement des clients télécoms (Customer Churn)**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-green)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/Tests-Pytest-success)](https://pytest.org/)

## 1. Présentation du projet

Ce projet consiste à développer une solution de Machine Learning permettant de prédire le risque de désabonnement (*Customer Churn*) des clients d'une entreprise de télécommunications.

L'objectif est d'identifier les clients susceptibles de résilier leur abonnement afin d'aider les équipes commerciales et marketing à anticiper les départs et à mettre en place des actions de fidélisation adaptées.

Le projet couvre plusieurs étapes d'un cycle de Machine Learning :

- Analyse exploratoire et contrôle de la qualité des données.
- Prétraitement et transformation des variables.
- Entraînement et comparaison de plusieurs modèles.
- Optimisation du seuil de classification.
- Interprétation des prédictions avec SHAP.
- Développement d'une API REST avec FastAPI.
- Tests automatisés avec Pytest.
- Versionnement du projet avec Git et GitHub.

## 2. Dataset

Le projet utilise le dataset **Telco Customer Churn**, qui contient des informations relatives aux clients et à leurs abonnements.

| Caractéristique | Valeur |
|---|---:|
| Nombre de lignes | 7 043 |
| Nombre de colonnes | 21 |
| Variable cible | Churn |
| Clients non désabonnés | 5 174 (73,46 %) |
| Clients désabonnés | 1 869 (26,54 %) |

La variable cible `Churn` indique si un client a quitté l'entreprise :

- `No` : le client est resté.
- `Yes` : le client a résilié son abonnement.

Le fichier source brut n'est pas inclus dans le dépôt GitHub. Il doit être obtenu séparément auprès de sa source d'origine, en respectant ses conditions d'utilisation.

## 3. Analyse et préparation des données

Une analyse de qualité a été effectuée avant l'entraînement des modèles.

Les principales opérations réalisées sont :

- Vérification des dimensions et des types de variables.
- Recherche des valeurs manquantes.
- Recherche des doublons.
- Vérification des identifiants clients.
- Contrôle des valeurs numériques.
- Analyse des valeurs aberrantes avec la méthode IQR.
- Analyse de la distribution de la variable cible.

### Traitement des valeurs manquantes

La variable `TotalCharges` contenait 11 valeurs manquantes. Ces observations correspondaient à des clients ayant une ancienneté (`tenure`) égale à zéro.

Ces valeurs ont été remplacées par zéro, conformément à l'hypothèse retenue pour ce projet.

### Prétraitement

La préparation des données comprend :

- Suppression de l'identifiant `customerID` des variables explicatives.
- Encodage de la variable cible `Churn` : No = 0 et Yes = 1.
- Encodage des variables catégorielles avec One-Hot Encoding.
- Standardisation des variables numériques.
- Séparation des données en ensembles d'entraînement et de test avec stratification.

| Ensemble | Nombre d'observations |
|---|---:|
| Entraînement | 5 634 |
| Test | 1 409 |

Après transformation, les 19 variables explicatives produisent 46 caractéristiques utilisables par les modèles.

## 4. Modélisation et comparaison

Trois algorithmes ont été entraînés et évalués :

1. Régression logistique (*Logistic Regression*).
2. Arbre de décision (*Decision Tree*).
3. Forêt aléatoire (*Random Forest*).

La comparaison repose notamment sur le Recall, le F1-score, le ROC-AUC et l'Average Precision.

### Résultats de validation

| Modèle | Accuracy | Precision | Recall | F1-score | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.7542 | 0.5241 | 0.7993 | 0.6331 | 0.8484 | **0.6666** |
| Decision Tree | 0.7161 | 0.4758 | 0.6890 | 0.5628 | 0.7823 | 0.5816 |
| Random Forest | **0.7684** | **0.5487** | 0.7157 | 0.6212 | 0.8397 | 0.6375 |

La régression logistique a été retenue sur la base de son meilleur score Average Precision en validation.

Ce choix est cohérent avec le déséquilibre de la variable cible et l'intérêt de mesurer la qualité de détection de la classe minoritaire.

## 5. Évaluation sur les données de test

Avec un seuil de classification de 0.50, le modèle final obtient :

| Indicateur | Résultat |
|---|---:|
| Accuracy | 73.81 % |
| Precision | 50.43 % |
| Recall | 78.34 % |
| F1-score | 61.36 % |
| ROC-AUC | 0.8415 |
| Average Precision | 0.6325 |

Matrice de confusion :

| | Prédit : Non | Prédit : Oui |
|---|---:|---:|
| Réel : Non | 747 | 288 |
| Réel : Oui | 81 | 293 |

Le modèle détecte ainsi une proportion importante des clients ayant effectivement résilié leur abonnement.

## 6. Optimisation du seuil de classification

Le seuil standard de 0.50 n'est pas nécessairement le meilleur choix pour une entreprise.

Une optimisation a été réalisée sur les données de validation en testant plusieurs seuils entre 0.20 et 0.80.

Le seuil de 0.65 a obtenu le meilleur F1-score sur la validation parmi les seuils testés.

### Comparaison des seuils sur le test

| Indicateur | Seuil 0.50 | Seuil 0.65 |
|---|---:|---:|
| Accuracy | 73.81 % | 77.64 % |
| Precision | 50.43 % | 56.72 % |
| Recall | 78.34 % | 66.58 % |
| F1-score | 61.36 % | 61.25 % |
| ROC-AUC | 0.8415 | 0.8415 |
| Average Precision | 0.6325 | 0.6325 |

**Interprétation :**

- Le seuil de 0.50 privilégie la détection des clients susceptibles de partir. Il permet d'obtenir un Recall plus élevé.
- Le seuil de 0.65 réduit les fausses alertes et améliore la précision, mais détecte moins de départs.

Le seuil de 0.65 a été retenu dans l'API. Il ne constitue toutefois pas un seuil universellement optimal : le choix dépend du coût commercial des faux positifs et des faux négatifs.

## 7. Interprétabilité avec SHAP

La méthode SHAP (*SHapley Additive exPlanations*) a été utilisée pour comprendre les variables qui influencent les prédictions du modèle.

Les principales variables selon leur importance globale moyenne sont :

| Variable | Importance SHAP moyenne |
|---|---:|
| Tenure | 1.016890 |
| MonthlyCharges | 0.589016 |
| TotalCharges | 0.404878 |
| InternetService – Fiber optic | 0.357878 |
| Contract – Month-to-month | 0.325134 |
| InternetService – DSL | 0.278440 |
| Contract – Two year | 0.256667 |

Les résultats mettent notamment en évidence l'importance de l'ancienneté, des charges mensuelles et du type de contrat dans les prédictions du modèle.

SHAP permet d'interpréter les associations utilisées par le modèle. Ces associations ne démontrent pas une relation de causalité.

Les visualisations générées comprennent :

- Importance globale des variables.
- Graphique SHAP Summary.
- Graphique SHAP Bar.
- Explication d'une prédiction individuelle avec Waterfall Plot.

## 8. Développement de l'API avec FastAPI

Une API REST a été développée afin de rendre le modèle accessible à travers des requêtes HTTP.

Elle charge :

- Le modèle de régression logistique optimisé.
- Le préprocesseur sauvegardé.
- Le seuil de classification retenu.

### Fonctionnalités

| Endpoint | Fonction |
|---|---|
| `GET /` | Présentation de l'API |
| `GET /health` | Vérification de disponibilité |
| `POST /predict` | Prédiction du risque de désabonnement |

### Exemple de réponse

```json
{
  "prediction": 1,
  "label": "Churn",
  "probability": 0.9084,
  "threshold": 0.65,
  "risk_level": "High"
}
```

La probabilité représente une estimation produite par le modèle, et non une certitude sur le comportement futur du client.

### Lancer l'API

Depuis la racine du projet :

```bash
uvicorn api:app --reload
```

Documentation interactive :

http://127.0.0.1:8000/docs

Vérification de l'état de l'API :

http://127.0.0.1:8000/health

## 9. Tests automatisés

Le projet utilise Pytest pour vérifier les fonctionnalités principales de l'API.

Les tests couvrent :

- La disponibilité de l'API.
- La route principale.
- Le fonctionnement de la prédiction.
- Le traitement des données invalides.

Exécution :

```bash
python -m pytest -v
```

Résultat obtenu :

```text
4 passed, 2 warnings
```

Les quatre tests automatisés ont réussi. Les avertissements affichés ne bloquent pas leur exécution.

## 10. Installation et exécution

### Prérequis

- Python 3.11
- Git
- Visual Studio Code (recommandé)

### Cloner le dépôt

```bash
git clone https://github.com/Gelasemi/telco-customer-churn-mlops.git
```

Accéder au dossier :

```bash
cd telco-customer-churn-mlops
```

### Créer un environnement virtuel

```bash
python -m venv .venv
```

Sous Windows avec Git Bash :

```bash
source .venv/Scripts/activate
```

### Installer les dépendances

```bash
python -m pip install -r requirements.txt
```

### Démarrer l'API

```bash
uvicorn api:app --reload
```

### Exécuter les tests

Dans un autre terminal, depuis la racine du projet :

```bash
python -m pytest -v
```

## 11. Structure du projet

```text
telco-customer-churn-mlops/
│
├── api.py
├── requirements.txt
├── .gitignore
│
├── preprocessing_results/
│   ├── preprocessor_telco.joblib
│   ├── feature_names.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   ├── train_final.csv
│   └── test_final.csv
│
├── model_results/
│   └── résultats et métriques des modèles
│
├── threshold_results/
│   ├── optimized_logistic_model.joblib
│   ├── selected_threshold.txt
│   ├── threshold_comparison.csv
│   ├── final_threshold_metrics.csv
│   └── visualisations
│
├── shap_results/
│   ├── shap_feature_importance.csv
│   ├── shap_summary.png
│   ├── shap_bar.png
│   ├── shap_waterfall_customer.png
│   └── shap_report.txt
│
├── results/
│   └── rapports de préparation des données
│
└── test/
    └── test_api.py
```

*Cette arborescence présente les principaux fichiers et dossiers de résultats du projet.*

## 12. Limites et perspectives d'amélioration

Ce projet constitue une démonstration technique d'un pipeline de prédiction du désabonnement.

Plusieurs améliorations sont envisageables :

- Tester des algorithmes supplémentaires, notamment XGBoost et LightGBM.
- Effectuer une validation croisée.
- Étudier la calibration des probabilités.
- Optimiser le seuil selon les coûts réels de fidélisation.
- Ajouter un suivi des performances du modèle.
- Mettre en place une surveillance du Data Drift et du Model Drift.
- Conteneuriser l'API avec Docker.
- Mettre en place un pipeline CI/CD.
- Déployer l'application dans un environnement cloud.

## 13. Conclusion

Ce projet démontre la mise en œuvre d'un processus complet de Machine Learning appliqué à la prédiction du désabonnement client.

Il combine analyse de données, préparation des variables, apprentissage supervisé, évaluation des performances, optimisation du seuil, interprétabilité SHAP et exposition du modèle à travers une API FastAPI.

Le modèle retenu, une régression logistique, atteint un ROC-AUC de **0.8415** et une Average Precision de **0.6325** sur le jeu de test.

Les résultats montrent également l'importance d'adapter le seuil de classification aux objectifs opérationnels de l'entreprise.

---

## Auteur

**Gelase Michelot**

Data Science | Machine Learning | AI | Data Analytics | MLOps

Madagascar | Remote Worldwide

- GitHub : https://github.com/Gelasemi
- Projet : https://github.com/Gelasemi/telco-customer-churn-mlops

