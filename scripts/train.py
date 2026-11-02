"""Treina o pipeline de churn e gera os artefatos consumidos pela API.

Reproduz, sem o notebook, as decisões documentadas em notebooks/01_treinamento.ipynb.

Uso (na raiz do projeto):
    python scripts/train.py

Saídas:
    models/model.joblib   -> pipeline treinado + metadados
    reports/metrics.json  -> métricas de validação e de teste
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Caminhos relativos à raiz do projeto (scripts/train.py -> parents[1])
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "telco_churn.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
METRICS_PATH = BASE_DIR / "reports" / "metrics.json"

RANDOM_STATE = 42
THRESHOLD = 0.5
TARGET = "Churn"
NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
CATEGORICAL = ["Partner", "Dependents", "InternetService", "OnlineSecurity",
               "TechSupport", "Contract", "PaperlessBilling", "PaymentMethod"]


def load_data(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Lê o CSV e separa variáveis explicativas e alvo."""
    df = pd.read_csv(path)
    # Espaços em branco viram NaN e são tratados pelo imputer do pipeline
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    X = df[NUMERIC + CATEGORICAL]
    y = (df[TARGET] == "Yes").astype(int)
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Imputação + escalonamento (numéricas) e imputação + one-hot (categóricas)."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC),
        ("cat", categorical, CATEGORICAL),
    ])


def candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                class_weight="balanced",
                                                random_state=RANDOM_STATE, n_jobs=-1),
    }


def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    proba = pipeline.predict_proba(X)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    return {
        "accuracy": round(accuracy_score(y, pred), 4),
        "precision": round(precision_score(y, pred), 4),
        "recall": round(recall_score(y, pred), 4),
        "f1": round(f1_score(y, pred), 4),
        "roc_auc": round(roc_auc_score(y, proba), 4),
    }


def main() -> None:
    X, y = load_data(DATA_PATH)

    # 70% treino, 15% validação, 15% teste — estratificado pelo alvo
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE)

    # Seleção do modelo pelo ROC-AUC na validação
    candidates = candidate_models()
    val_results = {}
    for name, estimator in candidates.items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)
        val_results[name] = evaluate(pipe, X_val, y_val)
        print(f"[validação] {name}: {val_results[name]}")
    best_name = max(val_results, key=lambda n: val_results[n]["roc_auc"])

    # Re-treino com treino + validação e avaliação única no teste
    final_pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("model", clone(candidates[best_name])),
    ])
    final_pipeline.fit(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))
    test_metrics = evaluate(final_pipeline, X_test, y_test)
    print(f"[teste] {best_name}: {test_metrics}")

    trained_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    model_version = f"churn-{best_name}-v1"

    # O artefato guarda o pipeline e os metadados que a API expõe
    bundle = {
        "pipeline": final_pipeline,
        "model_name": best_name,
        "model_version": model_version,
        "threshold": THRESHOLD,
        "feature_columns": NUMERIC + CATEGORICAL,
        "trained_at": trained_at,
        "sklearn_version": sklearn.__version__,
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    metrics = {
        "model_name": best_name,
        "model_version": model_version,
        "trained_at": trained_at,
        "threshold": THRESHOLD,
        "split": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "validation": val_results,
        "test": test_metrics,
    }
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Modelo salvo em {MODEL_PATH.relative_to(BASE_DIR)}")
    print(f"Métricas salvas em {METRICS_PATH.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()