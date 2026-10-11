from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "day2_training.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "xgboost_model.joblib"
ANOMALY_FILE = MODEL_DIR / "isolation_forest.joblib"
IMPUTER_FILE = MODEL_DIR / "sensor_imputer.joblib"
ENCODER_FILE = MODEL_DIR / "label_encoder.joblib"
EXPLAINER_FILE = MODEL_DIR / "shap_explainer.joblib"

FEATURES = ["temperature", "pressure", "vibration", "current", "rpm"]
TARGET = "state"


def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}\n"
            "Generate it with: python generatedata.py --samples 10000 "
            "--output data/day2_training.csv"
        )

    data = pd.read_csv(DATA_FILE)
    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError(f"CSV is missing these columns: {missing}")

    data = data[required].copy()

    for column in FEATURES:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data[TARGET] = data[TARGET].astype("string").str.strip()
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET] != ""]

    if len(data) < 10:
        raise ValueError("Please generate at least 10 valid data rows.")

    if data[TARGET].nunique() < 2:
        raise ValueError("The CSV must contain at least two different states.")

    return data


def train_system(data):
    X = data[FEATURES]
    y_text = data[TARGET].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_text)

    class_counts = pd.Series(y).value_counts()
    stratify_values = y if class_counts.min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_values,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_ready = imputer.fit_transform(X_train)
    X_test_ready = imputer.transform(X_test)

    classifier = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    classifier.fit(X_train_ready, y_train)

    predicted = classifier.predict(X_test_ready).astype(int)

    print("\nDATASET")
    print("-------")
    print(f"Total rows: {len(data)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows: {len(X_test)}")
    print(f"Features: {', '.join(FEATURES)}")
    print(f"States: {', '.join(encoder.classes_)}")

    print("\nXGBOOST EVALUATION")
    print("------------------")
    print(f"Accuracy: {accuracy_score(y_test, predicted):.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predicted,
            labels=np.arange(len(encoder.classes_)),
            target_names=encoder.classes_,
            zero_division=0,
        )
    )
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_test, predicted, labels=np.arange(len(encoder.classes_))))
    print("Class order:", list(encoder.classes_))

    anomaly_model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
    )
    anomaly_model.fit(X_train_ready)

    anomaly_predictions = anomaly_model.predict(X_test_ready)
    anomaly_count = int(np.sum(anomaly_predictions == -1))

    print("\nISOLATION FOREST EVALUATION")
    print("---------------------------")
    print(f"Unusual test readings: {anomaly_count} / {len(X_test_ready)}")
    print(f"Unusual reading rate: {anomaly_count / len(X_test_ready):.2%}")
    print("Note: Isolation Forest flags unusual patterns, not confirmed faults.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, MODEL_FILE)
    joblib.dump(anomaly_model, ANOMALY_FILE)
    joblib.dump(imputer, IMPUTER_FILE)
    joblib.dump(encoder, ENCODER_FILE)

    explainer = shap.TreeExplainer(classifier)
    joblib.dump(explainer, EXPLAINER_FILE)

    print(f"\nSaved models in: {MODEL_DIR}")
    return classifier, anomaly_model, imputer, encoder, explainer


def explain_prediction(model, explainer, row, encoder):
    print("\nSHAP EXPLANATION")
    print("----------------")
    try:
        shap_values = explainer.shap_values(row)

        if isinstance(shap_values, list):
            values = np.array(shap_values)
        else:
            values = np.asarray(shap_values)

        predicted_class = int(model.predict(row)[0])

        if values.ndim == 3:
            # Typical multiclass shape: samples x features x classes.
            feature_scores = values[0, :, predicted_class]
        elif values.ndim == 2:
            feature_scores = values[0]
        elif values.ndim == 1:
            feature_scores = values
        else:
            print("Could not interpret SHAP output for this version.")
            return

        ranking = sorted(
            zip(FEATURES, feature_scores),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )

        print("Features with the largest SHAP impact for the predicted class:")
        for feature, score in ranking:
            print(f"{feature}: {float(score):+.4f}")
        print("Positive/negative values indicate contribution direction in the model output.")
    except Exception as error:
        print(f"SHAP explanation could not be generated: {error}")
        print("The classifier prediction is still available.")


def predict_reading(model, anomaly_model, imputer, encoder, explainer):
    print("\nENTER SENSOR READINGS")
    print("Enter numeric values. Type q at any prompt to return to the menu.")

    values = []
    for feature in FEATURES:
        value = input(f"{feature}: ").strip()
        if value.lower() == "q":
            return
        try:
            values.append(float(value))
        except ValueError:
            print(f"{feature} must be a number.")
            return

    row = pd.DataFrame([values], columns=FEATURES)
    prepared = imputer.transform(row)

    prediction_id = int(model.predict(prepared)[0])
    probabilities = model.predict_proba(prepared)[0]
    state = encoder.inverse_transform([prediction_id])[0]
    anomaly_result = int(anomaly_model.predict(prepared)[0])
    anomaly_score = float(anomaly_model.decision_function(prepared)[0])

    print("\nPREDICTION")
    print("----------")
    print(f"Predicted state: {state}")
    print("\nClassifier scores:")
    for class_name, probability in sorted(
        zip(encoder.classes_, probabilities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{class_name}: {probability:.2%}")

    print("\nANOMALY DETECTION")
    print("-----------------")
    if anomaly_result == -1:
        print("Result: UNUSUAL sensor pattern detected")
    else:
        print("Result: No unusual pattern detected")
    print(f"Isolation Forest score: {anomaly_score:.4f}")
    print("This score is not a probability of machine failure.")

    explain_prediction(model, explainer, prepared, encoder)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    data = load_data()
    model, anomaly_model, imputer, encoder, explainer = train_system(data)

    while True:
        print("\nMACHINE MAINTENANCE SYSTEM")
        print("1. Predict a new sensor reading")
        print("2. Exit")
        choice = input("Choose an option: ").strip()

        if choice == "1":
            predict_reading(model, anomaly_model, imputer, encoder, explainer)
        elif choice == "2":
            print("Exiting system.")
            break
        else:
            print("Please enter 1 or 2.")


if __name__ == "__main__":
    main()
