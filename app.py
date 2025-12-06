from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

APP = Flask(__name__)

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "randomforest_model.joblib")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Place `randomforest_model.joblib` next to this script.")

model = joblib.load(MODEL_PATH)

# Default feature list — order used if model does not expose `feature_names_in_`.
DEFAULT_FEATURES = [
    "region",
    "resource_type",
    "water_type",
    "fauna",
    "passport_age",
    "is_old_passport",
    "technical_condition",
    "condition_inv",
    "water_level_upstream",
    "water_level_downstream",
    "level_diff",
    "exploitation_stress",
    "height",
    "width",
    "depth",
    "num_incidents_past",
    "latitude",
    "longitude",
]

if hasattr(model, "feature_names_in_"):
    FEATURE_ORDER = list(model.feature_names_in_)
else:
    FEATURE_ORDER = DEFAULT_FEATURES


def build_input_dataframe(json_data: dict) -> pd.DataFrame:
    """Build a single-row DataFrame in the correct feature order.

    Notes:
    - If a feature value is missing, a 400 error is returned by the caller.
    - Categorical/string values are left as-is; if the trained model expects encoded integers,
      the client should provide those codes (see README).
    """
    row = {}
    for feat in FEATURE_ORDER:
        if feat not in json_data:
            raise KeyError(f"Missing feature: {feat}")
        row[feat] = json_data[feat]

    df = pd.DataFrame([row], columns=FEATURE_ORDER)

    # Try to coerce numeric-like columns to numeric types where possible
    for col in df.columns:
        # skip if value already numeric
        if pd.api.types.is_numeric_dtype(df[col].dtype):
            continue
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            # leave as-is (e.g., categorical strings) — model may expect codes instead
            pass

    return df


@APP.route("/predict", methods=["POST"])
def predict():
    """Accepts JSON with the expected features and returns model prediction and probabilities.

    Example JSON body:
    {
      "region": 1,
      "resource_type": 0,
      "water_type": 2,
      ...
    }
    """
    try:
        payload = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON"}), 400

    if not isinstance(payload, dict):
        return jsonify({"error": "JSON body must be an object/dict with feature names"}), 400

    try:
        df = build_input_dataframe(payload)
    except KeyError as e:
        return jsonify({"error": str(e), "expected_features": FEATURE_ORDER}), 400
    except Exception as e:
        return jsonify({"error": f"Failed to build input: {str(e)}"}), 400

    try:
        pred = model.predict(df)
        # try predict_proba; if unavailable, set to None
        prob = None
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            # If binary, return probability for class 1
            if probs.shape[1] == 2:
                prob = float(probs[0, 1])
            else:
                prob = probs[0].tolist()

        result = {
            "prediction": int(pred[0]) if (hasattr(pred[0], "__int__")) else pred[0],
            "probability": prob,
            "used_features": FEATURE_ORDER,
        }
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {str(e)}"}), 500


@APP.route("/meta", methods=["GET"])
def meta():
    """Return information about the loaded model and expected features."""
    info = {
        "model_path": MODEL_PATH,
        "n_features_in": getattr(model, "n_features_in_", None),
        "feature_names": FEATURE_ORDER,
    }
    return jsonify(info)


if __name__ == "__main__":
    # Run with Flask's built-in server for local testing (not for production)
    APP.run(host="0.0.0.0", port=5000, debug=True)
