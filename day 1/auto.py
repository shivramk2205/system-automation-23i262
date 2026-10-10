
"""
23IF02 SYSTEM AUTOMATION
INTELLIGENT MULTI-MACHINE FAULT-BASED MAINTENANCE ENGINE

Features:
1. Continuous monitoring of 10 machines
2. Six simulated sensors per machine
3. Priority-based fault detection
4. Trend and recurring-fault analysis
5. Risk-based machine ranking
6. Automatic maintenance work-order creation
7. Complete event history
8. Consolidated sensor and decision logs
9. Persistent stop-state simulation
10. Maintenance status tracking
11. Append-only CSV logs

Run:
    python hands_on_starter.py

Stop:
    Ctrl+C

SIMULATION ONLY. Not a certified safety controller.
"""

import csv
import json
import logging
import random
import time
from collections import deque
from datetime import datetime
from pathlib import Path


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MACHINE_COUNT = 10
RUN_INTERVAL = 1
HISTORY_SIZE = 10

FLEET_LOG = BASE_DIR / "fleet_log.csv"
EVENT_LOG = BASE_DIR / "event_log.csv"
MAINTENANCE_LOG = BASE_DIR / "maintenance_log.csv"
TEXT_LOG = BASE_DIR / "fleet_events.log"
STATE_FILE = BASE_DIR / "fleet_state.json"

MACHINES = [
    f"MACHINE_{i:02d}"
    for i in range(1, MACHINE_COUNT + 1)
]

logging.basicConfig(
    filename=str(TEXT_LOG),
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    encoding="utf-8",
)

logger = logging.getLogger("FleetMaintenance")


# ---------------------------------------------------------------------------
# GENERAL UTILITIES
# ---------------------------------------------------------------------------

def timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def append_csv(path, fields, record):
    """Append a record and create the header when necessary."""

    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)

        if not exists or path.stat().st_size == 0:
            writer.writeheader()

        writer.writerow({
            field: record.get(field, "")
            for field in fields
        })


def log_event(machine, event_type, details, severity="INFO"):
    """Record significant events in both structured and text logs."""

    record = {
        "timestamp": timestamp(),
        "machine": machine,
        "event_type": event_type,
        "severity": severity,
        "details": details,
    }

    try:
        append_csv(
            EVENT_LOG,
            ["timestamp", "machine", "event_type",
             "severity", "details"],
            record,
        )

        logger.info(
            "%s | machine=%s | %s",
            event_type,
            machine,
            details,
        )
    except OSError:
        logger.exception("EVENT LOG WRITE FAILURE")
        print("WARNING: Could not write event log.")


# ---------------------------------------------------------------------------
# STEP 1 - SIMULATED INDUSTRIAL SENSOR DATA
# ---------------------------------------------------------------------------

def generate_sensor_data(machine, previous_fault=None):
    """
    Generate six simulated sensor readings.

    Faults can persist across cycles, making trend and persistence
    analysis more meaningful than independent random readings.
    """

    r = {
        "temperature": round(random.uniform(55, 72), 2),
        "pressure": round(random.uniform(4.0, 6.5), 2),
        "vibration": round(random.uniform(1.0, 3.5), 2),
        "current": round(random.uniform(8.0, 16.0), 2),
        "rpm": random.randint(1400, 2200),
        "coolant_flow": round(random.uniform(7.0, 12.0), 2),
    }

    # Continue an active simulated fault most of the time.
    fault = previous_fault

    if fault is None and random.random() < 0.06:
        fault = random.choice([
            "OVERHEATING",
            "HIGH_VIBRATION",
            "LOW_PRESSURE",
            "MOTOR_OVERLOAD",
            "COOLANT_FAILURE",
        ])

    # Occasionally allow a noncritical simulated fault to clear.
    if fault is not None and random.random() < 0.08:
        fault = None

    if fault == "OVERHEATING":
        r["temperature"] = round(random.uniform(82, 98), 2)

    elif fault == "HIGH_VIBRATION":
        r["vibration"] = round(random.uniform(5.5, 10), 2)

    elif fault == "LOW_PRESSURE":
        r["pressure"] = round(random.uniform(1.0, 2.8), 2)

    elif fault == "MOTOR_OVERLOAD":
        r["current"] = round(random.uniform(19, 29), 2)

    elif fault == "COOLANT_FAILURE":
        r["coolant_flow"] = round(random.uniform(0, 2.5), 2)
        r["temperature"] = round(random.uniform(78, 96), 2)

    r["scenario"] = fault or "NORMAL"
    return r, fault


# ---------------------------------------------------------------------------
# STEP 2 - INDUSTRIAL FAULT RULES
# ---------------------------------------------------------------------------

RULES = [
    {
        "name": "CRITICAL_OVERHEATING",
        "priority": 1,
        "severity": "CRITICAL",
        "risk": 98,
        "condition": lambda r: r["temperature"] >= 90,
        "action": "STOP_MACHINE",
        "maintenance": "Inspect cooling system, bearings and motor load",
    },
    {
        "name": "CRITICAL_VIBRATION",
        "priority": 2,
        "severity": "CRITICAL",
        "risk": 95,
        "condition": lambda r: r["vibration"] >= 8,
        "action": "STOP_MACHINE",
        "maintenance": "Inspect bearings, alignment and imbalance",
    },
    {
        "name": "CRITICAL_OVERCURRENT",
        "priority": 3,
        "severity": "CRITICAL",
        "risk": 94,
        "condition": lambda r: r["current"] >= 25,
        "action": "STOP_MACHINE",
        "maintenance": "Inspect motor load, wiring and drive",
    },
    {
        "name": "CRITICAL_PRESSURE",
        "priority": 4,
        "severity": "CRITICAL",
        "risk": 96,
        "condition": lambda r: r["pressure"] < 1.5,
        "action": "STOP_MACHINE",
        "maintenance": "Inspect pressure supply, leaks and pump",
    },
    {
        "name": "COOLANT_FAILURE",
        "priority": 5,
        "severity": "CRITICAL",
        "risk": 97,
        "condition": lambda r: (
            r["coolant_flow"] < 3
            and r["temperature"] >= 75
        ),
        "action": "STOP_MACHINE",
        "maintenance": "Inspect coolant pump, filter and flow sensor",
    },
    {
        "name": "HIGH_TEMPERATURE",
        "priority": 10,
        "severity": "WARNING",
        "risk": 65,
        "condition": lambda r: r["temperature"] >= 75,
        "action": "ACTIVATE_COOLING",
        "maintenance": "Inspect cooling performance",
    },
    {
        "name": "HIGH_VIBRATION",
        "priority": 11,
        "severity": "WARNING",
        "risk": 62,
        "condition": lambda r: r["vibration"] >= 5,
        "action": "MAINTENANCE_ALERT",
        "maintenance": "Check bearings and alignment",
    },
    {
        "name": "LOW_PRESSURE",
        "priority": 12,
        "severity": "WARNING",
        "risk": 55,
        "condition": lambda r: r["pressure"] < 3,
        "action": "REDUCE_LOAD",
        "maintenance": "Inspect pressure supply and possible leaks",
    },
    {
        "name": "HIGH_CURRENT",
        "priority": 13,
        "severity": "WARNING",
        "risk": 60,
        "condition": lambda r: r["current"] >= 18,
        "action": "REDUCE_LOAD",
        "maintenance": "Check electrical load and motor condition",
    },
    {
        "name": "ABNORMAL_RPM",
        "priority": 14,
        "severity": "WARNING",
        "risk": 50,
        "condition": lambda r: r["rpm"] < 800 or r["rpm"] > 2800,
        "action": "MAINTENANCE_ALERT",
        "maintenance": "Inspect speed control and motor drive",
    },
]


# ---------------------------------------------------------------------------
# STEP 3 - FAULT DETECTION
# ---------------------------------------------------------------------------

def evaluate_rules(reading):
    required = [
        "temperature", "pressure", "vibration",
        "current", "rpm", "coolant_flow",
    ]

    for sensor in required:
        value = reading.get(sensor)

        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
        ):
            return {
                "name": "INVALID_SENSOR_DATA",
                "priority": 0,
                "severity": "CRITICAL",
                "risk": 100,
                "action": "STOP_MACHINE",
                "maintenance": "Inspect sensor, wiring and data connection",
            }

    for rule in sorted(RULES, key=lambda x: x["priority"]):
        if rule["condition"](reading):
            return rule

    return {
        "name": "NORMAL_OPERATION",
        "priority": 999,
        "severity": "NORMAL",
        "risk": 5,
        "action": "CONTINUE_NORMAL",
        "maintenance": "",
    }


# ---------------------------------------------------------------------------
# STEP 4 - INTELLIGENT RISK ASSESSMENT
# ---------------------------------------------------------------------------

def calculate_risk(rule, reading, history, fault_duration,
                   unresolved_work_order):
    """
    Calculate an illustrative 0-100 maintenance priority index.

    Factors:
    - Severity of the current fault
    - Persistent abnormal readings
    - Worsening sensor trends
    - Unresolved maintenance work

    This score is NOT a calibrated probability of failure.
    """

    score = rule["risk"]
    reasons = []

    if rule["severity"] == "NORMAL":
        score = 5
        reasons.append("Normal readings")
        return score, reasons

    # Repeated fault persistence.
    if fault_duration >= 3:
        bonus = min(10, fault_duration)
        score += bonus
        reasons.append(f"Persistent fault: {fault_duration} cycles")

    # Compare recent readings with earlier readings.
    if len(history) >= 4:
        recent = list(history)
        earlier = recent[-4]
        current = recent[-1]

        for sensor in ("temperature", "vibration", "current"):
            old_value = earlier.get(sensor)
            new_value = current.get(sensor)

            if old_value is None or new_value is None:
                continue

            if new_value > old_value * 1.10:
                score += 3
                reasons.append(f"Rising {sensor}")

    # Unresolved work orders increase maintenance urgency.
    if unresolved_work_order:
        score += 5
        reasons.append("Maintenance work order unresolved")

    return min(100, score), reasons


# ---------------------------------------------------------------------------
# STEP 5 - SIMULATED ACTIONS
# ---------------------------------------------------------------------------

def take_action(machine, rule):
    messages = {
        "STOP_MACHINE":
            "Critical stop request; inspect before restart",
        "ACTIVATE_COOLING":
            "Cooling intervention recommended",
        "REDUCE_LOAD":
            "Load reduction recommended",
        "MAINTENANCE_ALERT":
            "Maintenance inspection recommended",
        "CONTINUE_NORMAL":
            "No fault detected",
        "MONITOR_ONLY":
            "Machine monitored; another machine has higher priority",
    }

    message = messages.get(rule["action"], "Review required")

    log_event(
        machine,
        "ACTION_SELECTED",
        f"Rule={rule['name']}; action={rule['action']}; {message}",
        rule["severity"],
    )

    return message


# ---------------------------------------------------------------------------
# STEP 6 - MAINTENANCE WORK-ORDER MANAGEMENT
# ---------------------------------------------------------------------------

def create_maintenance_order(machine, rule, risk, state):
    """
    Create one open maintenance record per machine and fault.

    Repeated cycles do not create duplicate work orders.
    """

    key = f"{machine}|{rule['name']}"

    if key in state["work_orders"]:
        return state["work_orders"][key]["order_id"]

    order_id = (
        f"WO-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        f"-{machine[-2:]}-{len(state['work_orders']) + 1}"
    )

    order = {
        "order_id": order_id,
        "machine": machine,
        "fault": rule["name"],
        "severity": rule["severity"],
        "risk": risk,
        "recommendation": rule["maintenance"],
        "status": "OPEN",
        "created_at": timestamp(),
    }

    state["work_orders"][key] = order

    append_csv(
        MAINTENANCE_LOG,
        [
            "timestamp", "order_id", "machine", "fault",
            "severity", "risk_score", "recommendation",
            "status", "event",
        ],
        {
            **order,
            "timestamp": timestamp(),
            "event": "WORK_ORDER_CREATED",
        },
    )

    log_event(
        machine,
        "MAINTENANCE_ORDER_CREATED",
        f"Order={order_id}; fault={rule['name']}; risk={risk}",
        rule["severity"],
    )

    print(
        f"\n  NEW MAINTENANCE ORDER: {order_id}\n"
        f"  Machine: {machine}\n"
        f"  Fault: {rule['name']}\n"
        f"  Recommendation: {rule['maintenance']}\n"
    )

    return order_id


def update_maintenance_status(state, machine, fault, new_status):
    """
    Utility for future operator integration.

    Example statuses: IN_PROGRESS, RESOLVED.
    Call this only after an authorized maintenance update.
    """

    key = f"{machine}|{fault}"
    order = state["work_orders"].get(key)

    if not order:
        return False

    order["status"] = new_status

    append_csv(
        MAINTENANCE_LOG,
        [
            "timestamp", "order_id", "machine", "fault",
            "severity", "risk_score", "recommendation",
            "status", "event",
        ],
        {
            **order,
            "timestamp": timestamp(),
            "event": f"STATUS_CHANGED_TO_{new_status}",
        },
    )

    log_event(
        machine,
        "MAINTENANCE_STATUS_CHANGED",
        f"Order={order['order_id']}; status={new_status}",
    )

    return True


# ---------------------------------------------------------------------------
# STEP 7 - PERSISTENT STATE
# ---------------------------------------------------------------------------

def save_state(state):
    """Save stop states, active faults and work orders across restarts."""

    data = {
        "machines": {
            machine: {
                "stopped": values["stopped"],
                "active_fault": values["active_fault"],
                "fault_duration": values["fault_duration"],
            }
            for machine, values in state["machines"].items()
        },
        "work_orders": state["work_orders"],
    }

    temporary = STATE_FILE.with_suffix(".tmp")

    try:
        temporary.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
        temporary.replace(STATE_FILE)
    except OSError:
        logger.exception("STATE SAVE FAILURE")
        print("WARNING: Could not persist fleet state.")


def load_state():
    state = {
        "machines": {
            machine: {
                "stopped": False,
                "active_fault": None,
                "fault_duration": 0,
                "history": deque(maxlen=HISTORY_SIZE),
            }
            for machine in MACHINES
        },
        "work_orders": {},
    }

    if not STATE_FILE.exists():
        return state

    try:
        saved = json.loads(STATE_FILE.read_text(encoding="utf-8"))

        for machine, values in saved.get("machines", {}).items():
            if machine in state["machines"]:
                state["machines"][machine].update({
                    "stopped": values.get("stopped", False),
                    "active_fault": values.get("active_fault"),
                    "fault_duration": values.get("fault_duration", 0),
                })

        state["work_orders"] = saved.get("work_orders", {})

    except (OSError, ValueError, TypeError):
        logger.exception("STATE LOAD FAILURE")
        print("WARNING: Saved state could not be loaded.")

    return state


# ---------------------------------------------------------------------------
# STEP 8 - CONSOLIDATED FLEET MONITORING
# ---------------------------------------------------------------------------

def main():
    state = load_state()
    tick = 0

    print("=" * 90)
    print(" INTELLIGENT MULTI-MACHINE FAULT-BASED MAINTENANCE")
    print(f" Machines: {MACHINE_COUNT} | Interval: {RUN_INTERVAL}s")
    print(" Logs: fleet_log.csv, event_log.csv, maintenance_log.csv")
    print(" Press Ctrl+C to stop")
    print("=" * 90)

    log_event(
        "FLEET",
        "SYSTEM_STARTED",
        f"Machines={MACHINE_COUNT}; interval={RUN_INTERVAL}s",
    )

    try:
        while True:
            start = time.monotonic()
            tick += 1
            results = []

            # Phase 1: Read and evaluate all machines.
            for machine in MACHINES:
                m = state["machines"][machine]

                reading, simulated_fault = generate_sensor_data(
                    machine, m["active_fault"]
                )

                rule = evaluate_rules(reading)
                previous_fault = m["active_fault"]

                # A latched stop remains critical until manually cleared.
                if m["stopped"]:
                    rule = {
                        **rule,
                        "name": "MACHINE_STOPPED_LATCHED",
                        "severity": "CRITICAL",
                        "risk": 100,
                        "action": "STOP_MACHINE",
                        "maintenance": "Authorized inspection and reset required",
                    }

                # Update persistent fault duration.
                current_fault = (
                    rule["name"]
                    if rule["severity"] != "NORMAL"
                    else None
                )

                if current_fault and current_fault == previous_fault:
                    m["fault_duration"] += 1
                elif current_fault:
                    m["fault_duration"] = 1
                else:
                    m["fault_duration"] = 0

                m["active_fault"] = current_fault

                if rule["action"] == "STOP_MACHINE":
                    if not m["stopped"]:
                        log_event(
                            machine,
                            "STOP_REQUEST",
                            f"Fault={rule['name']}",
                            "CRITICAL",
                        )
                    m["stopped"] = True

                m["history"].append(reading)

                # Only open work orders for actual faults.
                work_key = f"{machine}|{rule['name']}"
                order_open = (
                    work_key in state["work_orders"]
                    and state["work_orders"][work_key]["status"]
                    != "RESOLVED"
                )

                risk, reasons = calculate_risk(
                    rule,
                    reading,
                    m["history"],
                    m["fault_duration"],
                    order_open,
                )

                # Log fault transitions and create work orders once.
                if current_fault and current_fault != previous_fault:
                    log_event(
                        machine,
                        "FAULT_DETECTED",
                        f"Fault={current_fault}; risk={risk}",
                        rule["severity"],
                    )

                if (
                    current_fault
                    and rule["name"] != "MACHINE_STOPPED_LATCHED"
                    and not order_open
                ):
                    create_maintenance_order(
                        machine, rule, risk, state
                    )

                results.append({
                    "machine": machine,
                    "reading": reading,
                    "rule": rule,
                    "risk": risk,
                    "reasons": reasons,
                    "state": "STOPPED" if m["stopped"]
                    else "RUNNING" if rule["severity"] == "NORMAL"
                    else "WARNING",
                    "fault_duration": m["fault_duration"],
                })

            # Phase 2: Select highest-priority machine.
            # Every critical machine remains critical.
            results.sort(
                key=lambda x: (
                    x["rule"]["severity"] == "CRITICAL",
                    x["risk"],
                    x["fault_duration"],
                ),
                reverse=True,
            )

            target = results[0]

            print(f"\n{'=' * 90}")
            print(f"CYCLE {tick} | {timestamp()}")
            print("-" * 90)
            print(
                f"{'MACHINE':<14} {'TEMP':>7} {'VIB':>7} "
                f"{'CURRENT':>8} {'RISK':>6} "
                f"{'SEVERITY':<10} {'STATE':<8} FAULT"
            )

            for item in results:
                r = item["reading"]
                rule = item["rule"]

                marker = "  <<< PRIORITY" if (
                    item["machine"] == target["machine"]
                ) else ""

                print(
                    f"{item['machine']:<14} "
                    f"{r['temperature']:>7.1f} "
                    f"{r['vibration']:>7.2f} "
                    f"{r['current']:>8.2f} "
                    f"{item['risk']:>6} "
                    f"{rule['severity']:<10} "
                    f"{item['state']:<8} "
                    f"{rule['name']}{marker}"
                )

            # Phase 3: Think and act for the selected machine.
            print("\nCENTRAL MAINTENANCE DECISION")
            print(f"Machine: {target['machine']}")
            print(f"Fault: {target['rule']['name']}")
            print(f"Risk index: {target['risk']}/100")
            print(
                "Risk factors: "
                + (", ".join(target["reasons"]) or "Base rule severity")
            )

            outcome = take_action(
                target["machine"], target["rule"]
            )

            print(f"Selected action: {target['rule']['action']}")
            print(f"Outcome: {outcome}")

            log_event(
                "FLEET",
                "PRIORITY_SELECTED",
                f"Cycle={tick}; machine={target['machine']}; "
                f"fault={target['rule']['name']}; risk={target['risk']}",
                target["rule"]["severity"],
            )

            # Phase 4: Log every machine, not just the selected one.
            for item in results:
                r = item["reading"]
                rule = item["rule"]
                machine = item["machine"]

                is_selected = machine == target["machine"]

                record = {
                    "timestamp": timestamp(),
                    "cycle": tick,
                    "machine": machine,
                    "temperature": r["temperature"],
                    "pressure": r["pressure"],
                    "vibration": r["vibration"],
                    "current": r["current"],
                    "rpm": r["rpm"],
                    "coolant_flow": r["coolant_flow"],
                    "scenario": r["scenario"],
                    "fault": rule["name"],
                    "severity": rule["severity"],
                    "risk_score": item["risk"],
                    "fault_duration": item["fault_duration"],
                    "action": (
                        rule["action"] if is_selected
                        else "MONITOR_ONLY"
                    ),
                    "machine_state": item["state"],
                    "selected_for_attention": is_selected,
                }

                append_csv(
                    FLEET_LOG,
                    list(record.keys()),
                    record,
                )

            save_state(state)

            print("\nCYCLE LOGGING COMPLETE")
            print(f"Fleet log: {FLEET_LOG.name}")
            print(f"Event log: {EVENT_LOG.name}")
            print(f"Maintenance log: {MAINTENANCE_LOG.name}")
            print(f"Open work orders: {sum(1 for o in state['work_orders'].values() if o['status'] != 'RESOLVED')}")
            print("Monitoring continues...\n")

            elapsed = time.monotonic() - start
            time.sleep(max(0, RUN_INTERVAL - elapsed))

    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

    finally:
        save_state(state)
        log_event(
            "FLEET",
            "SYSTEM_STOPPED",
            f"Cycles completed={tick}",
        )

        print("\nSESSION SUMMARY")
        print(f"Cycles completed: {tick}")
        print(f"Fleet log: {FLEET_LOG}")
        print(f"Event log: {EVENT_LOG}")
        print(f"Maintenance log: {MAINTENANCE_LOG}")
        print(f"Persistent state: {STATE_FILE}")


if __name__ == "__main__":
    main()
