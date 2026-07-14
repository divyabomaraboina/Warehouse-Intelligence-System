# ── Imports ──────────────────────────────────────────────────
import os
import sys
import json
import joblib
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from src.features import load_featured_data

# ── Constants ─────────────────────────────────────────────────
MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

features = [
    'MONTH',
    'IS_SUMMER',
    'IS_HOLIDAY',
    'SUPPLIER_ENCODED',
    'ITEM_TYPE_ENCODED',
    'LAST_MONTH_SALES',
    'LAST_3_MONTHS_SALES'
]
TARGET = 'IS_HIGH_SALES'


# ── Functions ─────────────────────────────────────────────────

def prepare_data(train, test):
    X_train = train[features]
    y_train = train[TARGET]
    X_test  = test[features]
    y_test  = test[TARGET]
    return X_train, y_train, X_test, y_test


def train_baseline(X_train, y_train):
    scaler        = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    log_reg        = LogisticRegression(random_state=42, max_iter=1000)
    log_reg.fit(X_train_scaled, y_train)
    return log_reg, scaler


def train_xgboost(X_train, y_train):
    xgb_model = xgb.XGBClassifier(random_state=42)
    xgb_model.fit(X_train, y_train)
    return xgb_model


def evaluate_model(model, X_test, y_test, model_name):
    y_pred   = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n=== {model_name} Evaluation ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, y_pred))
    return accuracy


def plot_importance(model, feature_names):
    import matplotlib.pyplot as plt
    xgb.plot_importance(model, importance_type='weight', show_values=False)
    plt.title("XGBoost Feature Importance")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / "xgb_feature_importance.png")
    plt.show()


def save_model(model, model_name):
    path = MODELS_DIR / f"{model_name}.pkl"
    joblib.dump(model, path)
    print(f"Model saved: {path}")
    return path


# ── Entry point ───────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(Path(__file__).parent.parent)

    # load data
    train, test = load_featured_data()
    X_train, y_train, X_test, y_test = prepare_data(train, test)

    # train models
    logreg, scaler = train_baseline(X_train, y_train)
    xgb_model      = train_xgboost(X_train, y_train)

    # evaluate
    X_test_scaled  = scaler.transform(X_test)
    logreg_accuracy = evaluate_model(logreg, X_test_scaled, y_test, "Logistic Regression")
    xgb_accuracy    = evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # plot importance
    plot_importance(xgb_model, features)

    # save best model
    if xgb_accuracy > logreg_accuracy:
        save_model(xgb_model, "best_model")
    else:
        save_model(logreg, "best_model")

    # save metrics → dashboard + RAG use chesthay
    metrics = {
        "xgb_accuracy":      round(float(xgb_accuracy), 4),
        "logistic_accuracy": round(float(logreg_accuracy), 4),
        "train_size":        len(X_train),
        "test_size":         len(X_test),
        "best_model":        "XGBoost" if xgb_accuracy > logreg_accuracy else "Logistic Regression"
    }
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Metrics saved → models/metrics.json")