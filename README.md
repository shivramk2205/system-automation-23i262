# 🏭 Intelligent Industrial Predictive Maintenance System

An academic system-automation project that combines simulated industrial sensor readings, rule-based monitoring, machine learning, anomaly detection, explainable AI, and a Streamlit dashboard to support predictive maintenance decisions.

## ✨ Project Overview

Unexpected machine failures can cause downtime, higher maintenance costs, and production delays. This project simulates an industrial monitoring workflow and demonstrates how sensor readings can be analyzed to identify potential abnormal conditions and recommend maintenance actions.

The system combines:

- **Sensor data simulation** — generates temperature, pressure, vibration, current, RPM, and coolant-flow readings.
- **Rule-based monitoring** — checks sensor readings against predefined warning and critical thresholds.
- **XGBoost classification** — predicts a machine condition such as `NORMAL`, `OVERHEATING`, `HIGH_VIBRATION`, `LOW_PRESSURE`, or `MOTOR_OVERLOAD`.
- **Isolation Forest** — flags readings that look unusual compared with the data used to fit the anomaly detector.
- **SHAP explanations** — helps inspect which input features influenced a model prediction.
- **Interactive Streamlit dashboard** — displays sensor readings, predictions, anomaly status, rule results, and explanations.
- **Text and voice commands** — supports a small set of commands such as checking status, analyzing sensors, and requesting maintenance.
- **Demo role checks** — separates operator and admin actions for demonstration.
- **CSV logs** — records dashboard analyses and provides access to existing Day 1 monitoring and maintenance logs.

> **Important:** This is a student project and a software simulation. It does not connect to physical industrial equipment. Model predictions and synthetic data are for demonstration and experimentation, not for safety-critical decisions.

## 🧠 System Workflow

```text
Simulated Sensor Readings
          |
          v
   Input Preparation
          |
          +-------------------+
          |                   |
          v                   v
   Rule-Based Checks      ML Analysis
                          |         |
                          v         v
                       XGBoost  Isolation Forest
                          |         |
                          +----+----+
                               |
                               v
                       Decision Selection
                               |
                 +-------------+-------------+
                 |             |             |
                 v             v             v
              Continue     Maintenance    Review / Simulated
                              Alert          Stop
                 |
                 v
       SHAP Explanation + Dashboard
                 |
                 v
        CSV Event / Analysis Logs
```

Text and voice commands pass through intent matching and demo role validation before the dashboard accepts supported actions.

## 🧰 Technology Stack

- Python 3.12
- pandas and NumPy
- scikit-learn
- XGBoost
- SHAP
- joblib
- Streamlit
- `streamlit-mic-recorder` and `SpeechRecognition` for the optional voice interface
- CSV files for datasets and logs

## 📁 Project Structure

The project is organized into Day 1 rule-based automation and Day 2 machine-learning components. The tree below shows the main files; runtime logs and model artifacts may be created after running the scripts.

```text
system-automation-23i262/
├── day 1/
│   ├── auto.py
│   ├── automation.py
│   ├── hands_on_starter.py
│   ├── day1_log.csv
│   ├── day1_events.log
│   ├── event_log.csv
│   ├── fleet_log.csv
│   ├── fleet_events.log
│   ├── fleet_state.json
│   └── maintenance_log.csv
├── day2/
│   ├── data/
│   │   ├── day2_sample.csv
│   │   └── day2_training.csv
│   ├── models/
│   │   ├── xgboost_model.joblib
│   │   ├── isolation_forest.joblib
│   │   ├── label_encoder.joblib
│   │   ├── sensor_imputer.joblib
│   │   └── shap_explainer.joblib
│   ├── generatedata.py
│   ├── train.py
│   ├── final.py
│   ├── integrated_system.py
│   ├── dashboard.py
│   └── integrated_log.csv
├── README.md
└── requirements.txt
```

Your local folder names may differ slightly. For example, Windows Explorer may display `day 1` with a space, while the terminal commands must match the actual folder name exactly. Update paths in commands if your folder uses a different name.

## ⚙️ Setup

### 1. Install Python

Install Python 3.12 and make sure it is available in your terminal. Verify:

```powershell
python --version
```

If `python` is not recognized, use the full Python executable path or repair the Python PATH setting.

### 2. Clone the repository

```powershell
git clone https://github.com/shivramk2205/system-automation-23i262.git
cd system-automation-23i262
```

If you have already cloned the repository, open its folder in VS Code instead of cloning it again.

### 3. Create a virtual environment (recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

If PowerShell blocks activation, you can run the project using `.venv\Scripts\python.exe` without activating it.

### 4. Install dependencies

If `requirements.txt` contains the project dependencies:

```powershell
python -m pip install -r requirements.txt
```

If a package is missing, install the dependencies used by the Day 2 dashboard:

```powershell
python -m pip install pandas numpy scikit-learn xgboost joblib shap streamlit streamlit-mic-recorder SpeechRecognition
```

Voice transcription may require an internet connection. The text-command interface can still be used if voice transcription is unavailable.

## ▶️ Generate the Day 2 Dataset

Run the generator from the repository root, adjusting the path if your folder structure differs:

```powershell
python .\day2\generatedata.py --samples 10000 --output .\day2\data\day2_training.csv
```

To generate more rows, change `10000` to a larger number, for example `50000`:

```powershell
python .\day2\generatedata.py --samples 50000 --output .\day2\data\day2_training.csv
```

The generated dataset contains the sensor features and a `state` target label. It is synthetic data, so its labels and feature ranges reflect the generator's rules rather than measurements from real machinery.

## 🤖 Train the Models

Run the training script from the repository root:

```powershell
python .\day2\train.py
```

The training script expects the configured training CSV to exist at the path defined in `train.py`. If you receive a `FileNotFoundError`, check the `DATA_FILE` path near the top of that script and make sure it matches the actual location of `day2_training.csv`.

After successful training, the model artifacts should be available in `day2/models/`. The dashboard expects these model files:

- `xgboost_model.joblib`
- `isolation_forest.joblib`
- `sensor_imputer.joblib`
- `label_encoder.joblib`

The SHAP explainer artifact is optional if the dashboard creates an explainer at runtime. Model filenames must match the names used in `dashboard.py`.

## 📊 Launch the Dashboard

From the repository root:

```powershell
python -m streamlit run .\day2\dashboard.py
```

If you use the full Python 3.12 path on Windows, the equivalent command is:

```powershell
& "C:\Program Files\Python312\python.exe" -m streamlit run .\day2\dashboard.py
```

Streamlit prints a local URL in the terminal. Open that URL in your browser.

### Dashboard features

1. **Live monitoring** — generate a sensor reading and view the sensor values.
2. **Decision results** — inspect the XGBoost predicted state, model score, Isolation Forest anomaly result, rule result, severity, and final decision.
3. **SHAP explanation** — review the features with the largest reported impact on the prediction.
4. **Text commands** — try commands such as `show status`, `analyze sensors`, and `request maintenance`.
5. **Voice commands** — record a short command if the optional voice dependencies and microphone permissions are available.
6. **Security demonstration** — test the operator and admin role restrictions.
7. **Logs** — view and download the integrated Day 2 log and inspect available Day 1 fleet, event, and maintenance CSV files.

### Demo login

The dashboard currently uses local demo credentials:

| Role | Username | Password |
|---|---|---|
| Operator | `operator` | `operator123` |
| Admin | `admin` | `admin123` |

These credentials are hard-coded for demonstration. Do not use them for a public deployment or a real industrial environment.

## 🧪 Suggested Test Plan

- Generate several readings and check that the displayed values change.
- Confirm that XGBoost returns a condition label.
- Confirm that Isolation Forest returns an anomaly status.
- Inspect the rule severity and final decision for each reading.
- Review the SHAP feature impacts.
- Test a supported text command and, if configured, a voice command.
- Verify that the operator role cannot invoke admin-only commands.
- Open the logs tab and confirm that new dashboard analyses are written to the integrated log.
- Test the system with a separate testing dataset if available.

## 📄 Data and Model Notes

- **Features:** `temperature`, `pressure`, `vibration`, `current`, `rpm`
- **Target label:** `state`
- **Example labels:** `NORMAL`, `OVERHEATING`, `HIGH_VIBRATION`, `LOW_PRESSURE`, `MOTOR_OVERLOAD`
- Coolant flow may be displayed and checked by the rule engine, but the trained classifier uses the feature list configured in `train.py`.
- A high classifier score is the model's confidence for its predicted class, not the probability that a machine will fail.
- An Isolation Forest score is an anomaly score, not a failure probability.
- SHAP describes how features contribute to the model output; it does not prove physical causation.
- Synthetic test data can make metrics look better than real-world performance. Real sensor data and independent validation are needed before any operational use.

## 🔐 Security and Safety

This repository demonstrates basic role checks and command validation. It is not production authentication or industrial control security. The current action outputs are simulated. Do not connect this software to real machinery, emergency stops, or safety-critical actuators without a proper engineering, security, and safety review.

## 🚧 Current Scope and Future Improvements

- Connect the dashboard directly to the original Day 1 automation and work-order functions.
- Unify Day 1 and Day 2 event schemas.
- Add persistent machine identifiers and maintenance status updates.
- Improve authentication and store credentials securely.
- Add robust voice-command confirmation and better speech error handling.
- Validate models against real or independently collected sensor data.
- Add trend charts, fleet-level views, and configurable alert thresholds.
- Add automated tests for data validation, rules, permissions, and model predictions.

## 👨‍🎓 Project Context

Developed as a student system-automation project to explore the integration of rule-based automation, predictive maintenance models, anomaly detection, explainable AI, and an interactive dashboard.

## 📜 License

No license is specified yet. Add a `LICENSE` file if you want to define how others may use, modify, and distribute this project.
