import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# === PARAMETERS ===
DATA_PATH = "synthetic_gts_10000_improved.csv"
ARTIFACTS_DIR = "artifacts"
TARGET = "failure"
EXCLUDE_COLS = ["risk", "name"]

MODEL_PARAMS = {
    "n_estimators": 600,
    "max_depth": 18,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "max_features": "sqrt",
    "random_state": 42,
    "n_jobs": -1
}


# === FUNCTIONS ===
def load_and_prepare_data(path: str):
    """Load CSV and prepare features and target."""
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET] + EXCLUDE_COLS)
    y = df[TARGET]

    # Encode categorical columns
    for col in X.select_dtypes(include="object").columns:
        X[col] = X[col].astype("category").cat.codes

    return X, y, df


def train_model(X_train, y_train):
    """Train RandomForest model with given parameters."""
    model = RandomForestClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and print metrics."""
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print("\n🔥 MODEL METRICS")
    print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.4f}\n")

    print("Classification report:")
    print(classification_report(y_test, pred))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, pred))

    return pred, prob


def save_artifacts(model, feature_names, df_prepared):
    """Save model, feature importances, and prepared dataset."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Save model
    model_path = os.path.join(ARTIFACTS_DIR, "randomforest_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved → {model_path}")

    # Save feature importances
    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_
    }).sort_values(by="importance", ascending=False)
    fi_path = os.path.join(ARTIFACTS_DIR, "feature_importances.csv")
    fi.to_csv(fi_path, index=False)
    print(f"💾 Feature importances saved → {fi_path}")

    # Save prepared dataset
    df_ready_path = os.path.join(ARTIFACTS_DIR, "prepared_dataset.csv")
    df_prepared.to_csv(df_ready_path, index=False)
    print(f"💾 Prepared dataset saved → {df_ready_path}")


# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Load and prepare data
    X, y, df_prepared = load_and_prepare_data(DATA_PATH)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    evaluate_model(model, X_test, y_test)

    # Save artifacts
    save_artifacts(model, X.columns, df_prepared)