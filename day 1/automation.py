
"""
DAY 1 · HANDS-ON — Build Your Rule Engine
23IF02 System Automation

Industrial Automation Version

STEP 1  — Generate simulated sensor data
STEP 2  — Write automation rules
STEP 3  — Detect machine state from the rules
STEP 4  — Trigger an action
STEP 5  — Log every decision
STEP 6  — Add additional sensors
STEP 7  — Add custom rules
STEP 8  — Add priority rules

Features:
- Continuous monitoring every second
- Six simulated industrial sensors
- Priority-based automation rules
- Critical fault detection
- Simulated machine actions
- Complete CSV and text logging
- Continuous monitoring after a stop request
- Graceful exit using Ctrl+C

Run:
    python hands_on_starter.py

Simulation only: no physical machine is controlled.
"""

import random
import time
import csv
import logging
from datetime import datetime
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent

CSV_LOG_FILE = BASE_DIR / "day1_log.csv"
TEXT_LOG_FILE = BASE_DIR / "day1_events.log"

MACHINE_NAME = "INDUSTRIAL_MOTOR_01"
RUN_INTERVAL = 1 

logging.basicConfig(
    filename=str(TEXT_LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    encoding="utf-8",
)

logger = logging.getLogger("IndustrialRuleEngine")
def generate_sensor_data():
    """Generate simulated readings from industrial sensors."""

    # Normal operating readings
    temperature = round(random.uniform(55, 72), 2)
    pressure = round(random.uniform(4.0, 6.5), 2)
    vibration = round(random.uniform(1.0, 3.5), 2)
    current = round(random.uniform(8.0, 16.0), 2)
    rpm = random.randint(1400, 2200)
    coolant_flow = round(random.uniform(7.0, 12.0), 2)
    scenario = random.choices(
        [
            "NORMAL",
            "OVERHEATING",
            "HIGH_VIBRATION",
            "LOW_PRESSURE",
            "MOTOR_OVERLOAD",
            "COOLANT_FAILURE",
        ],
        weights=[60, 8, 8, 7, 9, 8],
        k=1,
    )[0]

    if scenario == "OVERHEATING":
        temperature = round(random.uniform(76, 98), 2)

    elif scenario == "HIGH_VIBRATION":
        vibration = round(random.uniform(5.0, 10.0), 2)

    elif scenario == "LOW_PRESSURE":
        pressure = round(random.uniform(1.0, 3.5), 2)

    elif scenario == "MOTOR_OVERLOAD":
        current = round(random.uniform(19.0, 29.0), 2)

    elif scenario == "COOLANT_FAILURE":
        coolant_flow = round(random.uniform(0.0, 2.5), 2)
        temperature = round(random.uniform(76, 95), 2)

    reading = {
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
        "current": current,
        "rpm": rpm,
        "coolant_flow": coolant_flow,
        "scenario": scenario,
    }

    return reading
RULES = [
    {
        "name": "EMERGENCY_OVERHEATING",
        "priority": 1,
        "severity": "CRITICAL",
        "condition": lambda r: r["temperature"] >= 90,
        "action": "STOP_MACHINE",
    },
    {
        "name": "EXCESSIVE_VIBRATION",
        "priority": 2,
        "severity": "CRITICAL",
        "condition": lambda r: r["vibration"] >= 8,
        "action": "STOP_MACHINE",
    },
    {
        "name": "MOTOR_OVERCURRENT",
        "priority": 3,
        "severity": "CRITICAL",
        "condition": lambda r: r["current"] >= 25,
        "action": "STOP_MACHINE",
    },
    {
        "name": "CRITICAL_LOW_PRESSURE",
        "priority": 4,
        "severity": "CRITICAL",
        "condition": lambda r: r["pressure"] < 1.5,
        "action": "STOP_MACHINE",
    },
    {
        "name": "COOLANT_FAILURE",
        "priority": 5,
        "severity": "CRITICAL",
        "condition": lambda r: (
            r["coolant_flow"] < 3
            and r["temperature"] >= 75
        ),
        "action": "STOP_MACHINE",
    },
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 10,
        "severity": "WARNING",
        "condition": lambda r: r["temperature"] >= 75,
        "action": "ACTIVATE_COOLING",
    },
    {
        "name": "HIGH_VIBRATION_WARNING",
        "priority": 11,
        "severity": "WARNING",
        "condition": lambda r: r["vibration"] >= 5,
        "action": "MAINTENANCE_ALERT",
    },
    {
        "name": "LOW_PRESSURE_WARNING",
        "priority": 12,
        "severity": "WARNING",
        "condition": lambda r: r["pressure"] < 3,
        "action": "REDUCE_LOAD",
    },
    {
        "name": "HIGH_CURRENT_WARNING",
        "priority": 13,
        "severity": "WARNING",
        "condition": lambda r: r["current"] >= 18,
        "action": "REDUCE_LOAD",
    },
    {
        "name": "ABNORMAL_RPM",
        "priority": 14,
        "severity": "WARNING",
        "condition": lambda r: r["rpm"] < 800 or r["rpm"] > 2800,
        "action": "MAINTENANCE_ALERT",
    },
]
def evaluate_rules(reading, rules):
    """
    Return rule name, action, and severity.

    The first matching rule in priority order is selected.
    Invalid sensor readings generate a critical simulated stop request.
    """

    required_sensors = [
        "temperature",
        "pressure",
        "vibration",
        "current",
        "rpm",
        "coolant_flow",
    ]

    for sensor in required_sensors:
        value = reading.get(sensor)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
        ):
            return (
                "INVALID_SENSOR_DATA",
                "STOP_MACHINE",
                "CRITICAL",
            )
        if value < 0:
            return (
                "INVALID_SENSOR_DATA",
                "STOP_MACHINE",
                "CRITICAL",
            )

    for rule in sorted(rules, key=lambda item: item["priority"]):
        try:
            if rule["condition"](reading):
                return (
                    rule["name"],
                    rule["action"],
                    rule["severity"],
                )
        except Exception:
            logger.exception(
                "Rule evaluation error: %s", rule.get("name")
            )
            return (
                "RULE_EVALUATION_ERROR",
                "STOP_MACHINE",
                "CRITICAL",
            )

    return "NO_RULE_FIRED", "CONTINUE_NORMAL", "NORMAL"

def take_action(action):
    """Describe the simulated response to an automation action."""

    actions = {
        "STOP_MACHINE": (
            "CRITICAL: Simulated stop request; "
            "inspection required before restart."
        ),
        "ACTIVATE_COOLING": (
            "Simulated cooling activation requested."
        ),
        "REDUCE_LOAD": (
            "Simulated load reduction requested."
        ),
        "MAINTENANCE_ALERT": (
            "Maintenance inspection recommended."
        ),
        "CONTINUE_NORMAL": (
            "Operating parameters are within configured limits."
        ),
    }

    outcome = actions.get(
        action,
        f"Unknown action: {action}",
    )

    logger.info(
        "ACTION | name=%s | outcome=%s",
        action,
        outcome,
    )

    return outcome

def log_decision(reading, rule_name, severity, action, outcome,
                 machine_state, tick):
    """Append every cycle to CSV and text event logs."""

    fields = [
        "timestamp",
        "machine",
        "cycle",
        "scenario",
        "temperature_C",
        "pressure_bar",
        "vibration_mm_s",
        "current_A",
        "rpm",
        "coolant_flow_L_min",
        "rule",
        "severity",
        "action",
        "outcome",
        "machine_state",
    ]

    timestamp = datetime.now().astimezone().isoformat(
        timespec="seconds"
    )

    record = {
        "timestamp": timestamp,
        "machine": MACHINE_NAME,
        "cycle": tick,
        "scenario": reading.get("scenario", "UNKNOWN"),
        "temperature_C": reading.get("temperature"),
        "pressure_bar": reading.get("pressure"),
        "vibration_mm_s": reading.get("vibration"),
        "current_A": reading.get("current"),
        "rpm": reading.get("rpm"),
        "coolant_flow_L_min": reading.get("coolant_flow"),
        "rule": rule_name,
        "severity": severity,
        "action": action,
        "outcome": outcome,
        "machine_state": machine_state,
    }

    try:
        # CSV log: one complete record per monitoring cycle.
        file_exists = CSV_LOG_FILE.exists()

        with CSV_LOG_FILE.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
            )

            if not file_exists or CSV_LOG_FILE.stat().st_size == 0:
                writer.writeheader()

            writer.writerow(record)

        # Text log: readable event history.
        logger.info(
            "CYCLE | %s",
            " | ".join(
                f"{key}={value}"
                for key, value in record.items()
            ),
        )

    except OSError:
        # Keep the error visible even if file logging fails.
        logger.exception("FAILED TO WRITE LOG FILE")
        print("ERROR: Could not write the log files.")


# ---------------------------------------------------------------------------
# STEP 6 — Additional sensors
# ---------------------------------------------------------------------------
# Implemented in generate_sensor_data():
# current, rpm, and coolant_flow.


# ---------------------------------------------------------------------------
# STEP 7 — Custom rules
# ---------------------------------------------------------------------------
# Implemented in RULES:
# COOLANT_FAILURE and ABNORMAL_RPM.


# ---------------------------------------------------------------------------
# STEP 8 — Priority rules
# ---------------------------------------------------------------------------
# Critical conditions are evaluated before warning conditions.


# ---------------------------------------------------------------------------
# MAIN AUTOMATION LOOP — Continuous operation
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print(" INDUSTRIAL AUTOMATION — CONTINUOUS RULE ENGINE")
    print(f" Machine       : {MACHINE_NAME}")
    print(" Monitoring    : Every second")
    print(" Logging       : CSV + text event log")
    print(" Mode          : SIMULATION ONLY")
    print(" Stop program  : Press Ctrl+C")
    print("=" * 75)

    logger.info(
        "SYSTEM_START | machine=%s | interval=%s seconds",
        MACHINE_NAME,
        RUN_INTERVAL,
    )

    tick = 0
    machine_stopped = False
    normal_count = 0
    warning_count = 0
    critical_count = 0
    previous_rule = None

    try:
        while True:
            cycle_start = time.monotonic()
            tick += 1

            try:
                # STEP 1: Generate sensor readings.
                reading = generate_sensor_data()

                # STEP 3: Evaluate the automation rules.
                rule_name, action, severity = evaluate_rules(
                    reading, RULES
                )

                # Maintain a latched simulated stop state.
                if machine_stopped:
                    rule_name = "MACHINE_STOPPED_LATCHED"
                    action = "STOP_MACHINE"
                    severity = "CRITICAL"
                    machine_state = "STOPPED"
                    outcome = (
                        "Stop state remains latched; "
                        "monitoring continues."
                    )
                else:
                    outcome = take_action(action)

                    if action == "STOP_MACHINE":
                        machine_stopped = True
                        machine_state = "STOPPED"
                    elif action == "CONTINUE_NORMAL":
                        machine_state = "RUNNING"
                    else:
                        machine_state = "WARNING"

                # Count every monitoring cycle by its severity.
                if severity == "NORMAL":
                    normal_count += 1
                elif severity == "WARNING":
                    warning_count += 1
                else:
                    critical_count += 1

                # Display the complete sensor and decision information.
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                print(f"\n[{timestamp}] CYCLE {tick}")
                print(f"Machine    : {MACHINE_NAME}")
                print(f"Scenario   : {reading['scenario']}")
                print(
                    f"Temperature: {reading['temperature']:.2f} °C"
                )
                print(f"Pressure   : {reading['pressure']:.2f} bar")
                print(
                    f"Vibration  : {reading['vibration']:.2f} mm/s"
                )
                print(f"Current    : {reading['current']:.2f} A")
                print(f"RPM        : {reading['rpm']}")
                print(
                    f"Coolant    : {reading['coolant_flow']:.2f} L/min"
                )
                print(f"Rule       : {rule_name}")
                print(f"Severity   : {severity}")
                print(f"Action     : {action}")
                print(f"Outcome    : {outcome}")
                print(f"State      : {machine_state}")

                # Log every reading, including normal cycles.
                log_decision(
                    reading,
                    rule_name,
                    severity,
                    action,
                    outcome,
                    machine_state,
                    tick,
                )

                # Log changes in the selected rule.
                if rule_name != previous_rule:
                    logger.info(
                        "RULE_CHANGE | previous=%s | current=%s",
                        previous_rule,
                        rule_name,
                    )
                    previous_rule = rule_name

                print(
                    f"Logged to   : {CSV_LOG_FILE.name}, "
                    f"{TEXT_LOG_FILE.name}"
                )

            except Exception:
                # A software-cycle error is logged and the loop continues.
                logger.exception(
                    "CYCLE_ERROR | cycle=%s", tick
                )
                print(
                    f"ERROR during cycle {tick}; "
                    "details written to the text log."
                )

            # Maintain approximately one-second cycle intervals.
            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0, RUN_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user.")

    finally:
        logger.info(
            "SYSTEM_STOP | cycles=%s | normal=%s | "
            "warnings=%s | critical=%s | stopped=%s",
            tick,
            normal_count,
            warning_count,
            critical_count,
            machine_stopped,
        )

        print("\n========== SESSION SUMMARY ==========")
        print(f"Machine             : {MACHINE_NAME}")
        print(f"Cycles processed    : {tick}")
        print(f"Normal cycles       : {normal_count}")
        print(f"Warning cycles      : {warning_count}")
        print(f"Critical cycles     : {critical_count}")
        print(
            f"Simulated state     : "
            f"{'STOPPED' if machine_stopped else 'RUNNING'}"
        )
        print(f"CSV log             : {CSV_LOG_FILE}")
        print(f"Text event log      : {TEXT_LOG_FILE}")
        print("=====================================")


if __name__ == "__main__":
    main()
