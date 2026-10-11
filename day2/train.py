from pathlib import Path

import joblib
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "day2_sample.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURES = ["temperature", "pressure", "vibration", "current", "rpm"]
TARGET = "state"


def main():
    data = pd.read_csv(DATA_FILE)

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    data[TARGET] = data[TARGET].astype("string").str.strip()
    data = data.dropna(subset=[TARGET])
    data = data[data[TARGET] != ""]

    X = data[FEATURES]
    labels = data[TARGET].astype(str)

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    imputer = SimpleImputer(strategy="median")
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    xgb_model = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.08,
        objective="multi:softprob",
        num_class=len(encoder.classes_),
        eval_metric="mlogloss",
        random_state=42
    )

    xgb_model.fit(X_train, y_train)
    predictions = xgb_model.predict(X_test)

    print("\nXGBOOST EVALUATION")
    print("------------------")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2%}")
    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        labels=range(len(encoder.classes_)),
        target_names=encoder.classes_,
        zero_division=0
    ))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    isolation_model = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42
    )

    isolation_model.fit(X_train)
    anomaly_predictions = isolation_model.predict(X_test)
    anomaly_count = (anomaly_predictions == -1).sum()

    print("\nISOLATION FOREST EVALUATION")
    print("---------------------------")
    print(f"Test rows: {len(X_test)}")
    print(f"Unusual readings: {anomaly_count}")
    print(f"Unusual reading rate: {anomaly_count / len(X_test):.2%}")
    print("1 means normal; -1 means unusual.")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODEL_DIR / "xgboost_model.joblib")
    joblib.dump(isolation_model, MODEL_DIR / "isolation_forest.joblib")
    joblib.dump(imputer, MODEL_DIR / "sensor_imputer.joblib")
    joblib.dump(encoder, MODEL_DIR / "label_encoder.joblib")

    print(f"\nModels saved in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
