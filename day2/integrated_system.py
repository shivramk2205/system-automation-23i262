from pathlib import Path
from datetime import datetime
import csv
import random
import time

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
LOG_FILE = BASE_DIR / "integrated_log.csv"

FEATURES = ["temperature", "pressure", "vibration", "current", "rpm"]

xgb_model = joblib.load(MODEL_DIR / "xgboost_model.joblib")
isolation_model = joblib.load(MODEL_DIR / "isolation_forest.joblib")
imputer = joblib.load(MODEL_DIR / "sensor_imputer.joblib")
encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")


def generate_sensor_data():
    temperature = round(random.uniform(55, 72), 2)
    pressure = round(random.uniform(4.0, 6.5), 2)
    vibration = round(random.uniform(1.0, 3.5), 2)
    current = round(random.uniform(8.0, 16.0), 2)
    rpm = random.randint(1400, 2200)
    coolant_flow = round(random.uniform(4.0, 8.0), 2)

    state = random.choices(
        ["NORMAL", "OVERHEATING", "HIGH_VIBRATION",
         "LOW_PRESSURE", "MOTOR_OVERLOAD"],
        weights=[60, 10, 10, 10, 10],
        k=1
    )[0]

    if state == "OVERHEATING":
        temperature = round(random.uniform(76, 98), 2)
    elif state == "HIGH_VIBRATION":
        vibration = round(random.uniform(5.0, 10.0), 2)
    elif state == "LOW_PRESSURE":
        pressure = round(random.uniform(1.0, 3.5), 2)
    elif state == "MOTOR_OVERLOAD":
        current = round(random.uniform(19.0, 29.0), 2)

    return {
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
        "current": current,
        "rpm": rpm,
        "coolant_flow": coolant_flow
    }


def evaluate_rules(reading):
    if reading["temperature"] >= 90:
        return "CRITICAL_OVERHEATING", "SIMULATED_STOP"
    if reading["vibration"] >= 8:
        return "CRITICAL_VIBRATION", "SIMULATED_STOP"
    if reading["current"] >= 25:
        return "CRITICAL_OVERCURRENT", "SIMULATED_STOP"
    if reading["pressure"] < 1.5:
        return "CRITICAL_PRESSURE", "SIMULATED_STOP"
    if reading["coolant_flow"] < 3 and reading["temperature"] >= 75:
        return "COOLANT_FAILURE", "SIMULATED_STOP"
    if reading["temperature"] >= 75:
        return "HIGH_TEMPERATURE", "WARNING"
    if reading["vibration"] >= 5:
        return "HIGH_VIBRATION", "WARNING"
    if reading["pressure"] < 3:
        return "LOW_PRESSURE", "WARNING"
    if reading["current"] >= 18:
        return "HIGH_CURRENT", "WARNING"
    return "NORMAL", "CONTINUE"


def predict_reading(reading):
    row = pd.DataFrame(
        [[reading[feature] for feature in FEATURES]],
        columns=FEATURES
    )
    row_ready = imputer.transform(row)

    prediction = int(xgb_model.predict(row_ready)[0])
    state = encoder.inverse_transform([prediction])[0]

    anomaly = isolation_model.predict(row_ready)[0]
    return state, "UNUSUAL" if anomaly == -1 else "NORMAL"


def write_log(reading, predicted_state, anomaly, rule, action):
    fields = [
        "timestamp", "temperature", "pressure", "vibration",
        "current", "rpm", "coolant_flow", "predicted_state",
        "anomaly", "rule", "action"
    ]

    exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        if not exists:
            writer.writeheader()

        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            **reading,
            "predicted_state": predicted_state,
            "anomaly": anomaly,
            "rule": rule,
            "action": action
        })


def main():
    print("INTEGRATED PREDICTIVE MAINTENANCE SYSTEM")
    print("Press Ctrl+C to stop monitoring.\n")

    try:
        while True:
            reading = generate_sensor_data()
            predicted_state, anomaly = predict_reading(reading)
            rule, action = evaluate_rules(reading)

            if anomaly == "UNUSUAL" and action == "CONTINUE":
                action = "REVIEW_REQUIRED"

            print("-" * 55)
            print("Sensor readings:", reading)
            print("ML prediction:", predicted_state)
            print("Anomaly status:", anomaly)
            print("Rule result:", rule)
            print("Decision:", action)

            write_log(reading, predicted_state, anomaly, rule, action)
            time.sleep(2)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")
        print("Log saved to:", LOG_FILE)


if __name__ == "__main__":
    main()