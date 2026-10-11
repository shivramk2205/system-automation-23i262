from pathlib import Path
from datetime import datetime
import csv
import random

import joblib
import pandas as pd
import streamlit as st

try:
    import shap
except ImportError:
    shap = None

try:
    from streamlit_mic_recorder import mic_recorder
except ImportError:
    mic_recorder = None


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_DIR = BASE_DIR / "models"
LOG_FILE = BASE_DIR / "integrated_log.csv"
FLEET_LOG = PROJECT_DIR / "fleet_log.csv"
EVENT_LOG = PROJECT_DIR / "event_log.csv"
MAINTENANCE_LOG = PROJECT_DIR / "maintenance_log.csv"

FEATURES = ["temperature", "pressure", "vibration", "current", "rpm"]
MODEL_PATHS = {
    "xgb": MODEL_DIR / "xgboost_model.joblib",
    "iforest": MODEL_DIR / "isolation_forest.joblib",
    "imputer": MODEL_DIR / "sensor_imputer.joblib",
    "encoder": MODEL_DIR / "label_encoder.joblib",
    "explainer": MODEL_DIR / "shap_explainer.joblib",
}

st.set_page_config(page_title="Industrial Predictive Maintenance", page_icon="🏭", layout="wide")


def load_models():
    missing = [str(path) for key, path in MODEL_PATHS.items()
               if key != "explainer" and not path.exists()]
    if missing:
        st.error("Missing model files. Run the Day 2 training script first:\n\n" + "\n".join(missing))
        st.stop()

    models = {
        "xgb": joblib.load(MODEL_PATHS["xgb"]),
        "iforest": joblib.load(MODEL_PATHS["iforest"]),
        "imputer": joblib.load(MODEL_PATHS["imputer"]),
        "encoder": joblib.load(MODEL_PATHS["encoder"]),
        "explainer": None,
    }

    if MODEL_PATHS["explainer"].exists():
        try:
            models["explainer"] = joblib.load(MODEL_PATHS["explainer"])
        except Exception:
            models["explainer"] = None
    return models


def generate_sensor_data():
    temperature = round(random.uniform(55, 72), 2)
    pressure = round(random.uniform(4.0, 6.5), 2)
    vibration = round(random.uniform(1.0, 3.5), 2)
    current = round(random.uniform(8.0, 16.0), 2)
    rpm = random.randint(1400, 2200)
    coolant_flow = round(random.uniform(4.0, 8.0), 2)

    state = random.choices(
        ["NORMAL", "OVERHEATING", "HIGH_VIBRATION", "LOW_PRESSURE", "MOTOR_OVERLOAD"],
        weights=[60, 10, 10, 10, 10],
        k=1,
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
        "coolant_flow": coolant_flow,
    }


def evaluate_rules(r):
    if r["temperature"] >= 90:
        return "CRITICAL_OVERHEATING", "CRITICAL", "SIMULATED_STOP"
    if r["vibration"] >= 8:
        return "CRITICAL_VIBRATION", "CRITICAL", "SIMULATED_STOP"
    if r["current"] >= 25:
        return "CRITICAL_OVERCURRENT", "CRITICAL", "SIMULATED_STOP"
    if r["pressure"] < 1.5:
        return "CRITICAL_PRESSURE", "CRITICAL", "SIMULATED_STOP"
    if r["coolant_flow"] < 3 and r["temperature"] >= 75:
        return "COOLANT_FAILURE", "CRITICAL", "SIMULATED_STOP"
    if r["temperature"] >= 75:
        return "HIGH_TEMPERATURE", "WARNING", "MAINTENANCE_ALERT"
    if r["vibration"] >= 5:
        return "HIGH_VIBRATION", "WARNING", "MAINTENANCE_ALERT"
    if r["pressure"] < 3:
        return "LOW_PRESSURE", "WARNING", "MAINTENANCE_ALERT"
    if r["current"] >= 18:
        return "HIGH_CURRENT", "WARNING", "MAINTENANCE_ALERT"
    if r["rpm"] < 800 or r["rpm"] > 2800:
        return "ABNORMAL_RPM", "WARNING", "MAINTENANCE_ALERT"
    return "NORMAL_OPERATION", "NORMAL", "CONTINUE"


def analyze_reading(reading, models):
    row = pd.DataFrame([[reading[f] for f in FEATURES]], columns=FEATURES)
    prepared = models["imputer"].transform(row)

    class_id = int(models["xgb"].predict(prepared)[0])
    predicted_state = str(models["encoder"].inverse_transform([class_id])[0])
    probabilities = models["xgb"].predict_proba(prepared)[0]
    confidence = float(max(probabilities))

    anomaly_result = int(models["iforest"].predict(prepared)[0])
    anomaly_score = float(models["iforest"].decision_function(prepared)[0])
    rule, severity, rule_action = evaluate_rules(reading)

    if severity == "CRITICAL":
        decision = "SIMULATED_STOP"
    elif anomaly_result == -1:
        decision = "REVIEW_REQUIRED"
    elif severity == "WARNING" or predicted_state != "NORMAL":
        decision = "MAINTENANCE_ALERT"
    else:
        decision = "CONTINUE"

    return {
        "predicted_state": predicted_state,
        "confidence": confidence,
        "anomaly": "UNUSUAL" if anomaly_result == -1 else "NORMAL",
        "anomaly_score": anomaly_score,
        "rule": rule,
        "severity": severity,
        "rule_action": rule_action,
        "decision": decision,
    }


def explain_reading(reading, models):
    row = pd.DataFrame([[reading[f] for f in FEATURES]], columns=FEATURES)
    prepared = models["imputer"].transform(row)
    model = models["xgb"]

    try:
        if shap is not None:
            explainer = models["explainer"] or shap.TreeExplainer(model)
            values = explainer.shap_values(prepared)
            class_id = int(model.predict(prepared)[0])

            if isinstance(values, list):
                scores = values[class_id][0]
            else:
                values = getattr(values, "values", values)
                values = __import__("numpy").asarray(values)
                if values.ndim == 3:
                    if values.shape[1] == len(FEATURES):
                        scores = values[0, :, class_id]
                    else:
                        scores = values[class_id, 0, :]
                elif values.ndim == 2:
                    scores = values[0]
                else:
                    scores = values.reshape(-1)[:len(FEATURES)]

            explain = pd.DataFrame({
                "Feature": FEATURES,
                "SHAP impact": [float(x) for x in scores],
            })
            explain["Absolute impact"] = explain["SHAP impact"].abs()
            return explain.sort_values("Absolute impact", ascending=False), None
    except Exception as error:
        return None, str(error)

    return None, "SHAP is not installed. Install it with: python -m pip install shap"


def append_log(reading, result, source, role):
    record = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source": source,
        "role": role,
        **reading,
        **result,
    }
    exists = LOG_FILE.exists()
    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(record.keys()))
        if not exists or LOG_FILE.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(record)


def interpret_command(command):
    text = command.lower().strip()
    if any(word in text for word in ["status", "dashboard", "show machines", "show status"]):
        return "show_status"
    if any(word in text for word in ["predict", "check sensors", "analyze", "run prediction"]):
        return "analyze"
    if any(word in text for word in ["maintenance", "service", "inspect"]):
        return "maintenance"
    if any(word in text for word in ["stop", "shutdown", "turn off"]):
        return "stop"
    return "unknown"


def validate_action(intent, role):
    allowed = {
        "operator": {"show_status", "analyze", "maintenance"},
        "admin": {"show_status", "analyze", "maintenance", "stop"},
    }
    if intent == "unknown":
        return False, "Command not recognized."
    if intent not in allowed.get(role, set()):
        return False, f"Security blocked this command for role '{role}'."
    return True, "Command passed the demo role and action checks."


def login():
    st.sidebar.subheader("Demo login")
    st.sidebar.caption("Demo only. Do not use these credentials for a deployed system.")

    if "role" not in st.session_state:
        st.session_state.role = None

    if st.session_state.role is None:
        with st.sidebar.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            users = {
                "operator": ("operator123", "operator"),
                "admin": ("admin123", "admin"),
            }
            if username in users and password == users[username][0]:
                st.session_state.role = users[username][1]
                st.session_state.username = username
                st.rerun()
            st.sidebar.error("Login failed. Check username and password.")
        st.info("Sign in with the local demo account to open the dashboard.")
        st.stop()

    st.sidebar.success(f"{st.session_state.username} ({st.session_state.role})")
    if st.sidebar.button("Sign out"):
        st.session_state.role = None
        st.session_state.username = None
        st.rerun()
    return st.session_state.role


def show_csv(path, title):
    st.subheader(title)
    if path.exists():
        try:
            data = pd.read_csv(path)
            st.caption(f"Source: {path.name} · {len(data)} rows")
            st.dataframe(data.tail(100), use_container_width=True)
        except Exception as error:
            st.warning(f"Could not read {path.name}: {error}")
    else:
        st.info(f"{path.name} has not been created yet.")


def main():
    st.title("🏭 Intelligent Industrial Predictive Maintenance")
    st.caption("Sensor simulation · Rules · XGBoost · Isolation Forest · SHAP · NLP · Demo security")

    role = login()
    models = load_models()

    if "latest_reading" not in st.session_state:
        st.session_state.latest_reading = generate_sensor_data()
        st.session_state.latest_result = analyze_reading(st.session_state.latest_reading, models)

    sensor_tab, command_tab, logs_tab = st.tabs(["Live monitoring", "Click / voice commands", "Day 1 + Day 2 logs"])

    with sensor_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Sensor readings")
            if st.button("Generate new reading", type="primary"):
                st.session_state.latest_reading = generate_sensor_data()
                st.session_state.latest_result = analyze_reading(st.session_state.latest_reading, models)
                append_log(st.session_state.latest_reading, st.session_state.latest_result, "dashboard_button", role)
            reading = st.session_state.latest_reading
            cols = st.columns(3)
            cols[0].metric("Temperature", f"{reading['temperature']:.2f} °C")
            cols[1].metric("Pressure", f"{reading['pressure']:.2f} bar")
            cols[2].metric("Vibration", f"{reading['vibration']:.2f}")
            cols[0].metric("Current", f"{reading['current']:.2f} A")
            cols[1].metric("RPM", reading["rpm"])
            cols[2].metric("Coolant flow", f"{reading['coolant_flow']:.2f}")

        with right:
            st.subheader("Decision results")
            result = st.session_state.latest_result
            st.metric("Predicted state", result["predicted_state"])
            st.metric("Model score", f"{result['confidence']:.1%}")
            st.metric("Anomaly status", result["anomaly"])
            st.write("Rule:", result["rule"])
            st.write("Severity:", result["severity"])
            st.write("Suggested rule action:", result["rule_action"])
            st.write("Final decision:", result["decision"])
            st.caption(f"Isolation Forest score: {result['anomaly_score']:.4f}; not a failure probability.")

        st.divider()
        st.subheader("SHAP explanation")
        explanation, error = explain_reading(st.session_state.latest_reading, models)
        if explanation is not None:
            st.write("Features with the largest impact on the current prediction:")
            st.dataframe(explanation, hide_index=True, use_container_width=True)
            st.caption("SHAP impacts explain model output; they are not proof of physical causation.")
        else:
            st.warning(error)

    with command_tab:
        st.subheader("Text command")
        command = st.text_input("Type a command", placeholder="show status / analyze sensors / request maintenance")
        if st.button("Process text command"):
            intent = interpret_command(command)
            allowed, message = validate_action(intent, role)
            st.write("Detected intent:", intent)
            st.write("Security:", "APPROVED" if allowed else "REJECTED")
            st.info(message)
            if allowed and intent == "show_status":
                st.json({"reading": st.session_state.latest_reading, "result": st.session_state.latest_result})
            elif allowed and intent == "analyze":
                st.session_state.latest_result = analyze_reading(st.session_state.latest_reading, models)
                append_log(st.session_state.latest_reading, st.session_state.latest_result, "text_command", role)
                st.success("Reading analyzed and logged.")
            elif allowed and intent == "maintenance":
                result = dict(st.session_state.latest_result)
                result["decision"] = "MAINTENANCE_REQUESTED"
                append_log(st.session_state.latest_reading, result, "text_command", role)
                st.success("Simulated maintenance request logged.")
            elif allowed and intent == "stop":
                result = dict(st.session_state.latest_result)
                result["decision"] = "SIMULATED_STOP_REQUESTED"
                append_log(st.session_state.latest_reading, result, "text_command", role)
                st.warning("Simulated stop request logged. No physical equipment was controlled.")

        st.divider()
        st.subheader("Voice command")
        if mic_recorder is None:
            st.info("Install voice support in the terminal using the commands below, then restart the dashboard.")
        else:
            audio = mic_recorder(start_prompt="Start recording", stop_prompt="Stop recording", key="voice_recorder")
            if audio and audio.get("bytes"):
                st.audio(audio["bytes"], format="audio/wav")
                try:
                    import speech_recognition as sr
                    import io
                    import wave
                    recognizer = sr.Recognizer()
                    with sr.AudioFile(io.BytesIO(audio["bytes"])) as source:
                        voice_audio = recognizer.record(source)
                    voice_text = recognizer.recognize_google(voice_audio)
                    st.write("Recognized command:", voice_text)
                    intent = interpret_command(voice_text)
                    allowed, message = validate_action(intent, role)
                    st.write("Detected intent:", intent)
                    st.write("Security:", "APPROVED" if allowed else "REJECTED")
                    st.info(message)
                    if allowed and intent == "show_status":
                        st.json({"reading": st.session_state.latest_reading, "result": st.session_state.latest_result})
                    elif allowed and intent == "analyze":
                        result = analyze_reading(st.session_state.latest_reading, models)
                        st.session_state.latest_result = result
                        append_log(st.session_state.latest_reading, result, "voice_command", role)
                        st.success("Reading analyzed and logged.")
                    elif allowed and intent == "maintenance":
                        result = dict(st.session_state.latest_result)
                        result["decision"] = "MAINTENANCE_REQUESTED"
                        append_log(st.session_state.latest_reading, result, "voice_command", role)
                        st.success("Simulated maintenance request logged.")
                    elif allowed and intent == "stop":
                        result = dict(st.session_state.latest_result)
                        result["decision"] = "SIMULATED_STOP_REQUESTED"
                        append_log(st.session_state.latest_reading, result, "voice_command", role)
                        st.warning("Simulated stop request logged.")
                except Exception as error:
                    st.error(f"Voice transcription failed: {error}. Try the text command instead.")

    with logs_tab:
        st.subheader("Integrated Day 2 log")
        show_csv(LOG_FILE, "Current dashboard log")
        st.divider()
        show_csv(FLEET_LOG, "Day 1 fleet monitoring log")
        show_csv(EVENT_LOG, "Day 1 event log")
        show_csv(MAINTENANCE_LOG, "Day 1 maintenance work orders")

        if LOG_FILE.exists():
            st.download_button(
                "Download Day 2 integrated log",
                data=LOG_FILE.read_bytes(),
                file_name="integrated_log.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
