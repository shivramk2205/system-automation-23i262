
import argparse
import csv
import random
from collections import Counter
from pathlib import Path


FEATURES = [
    "temperature",
    "pressure",
    "vibration",
    "current",
    "rpm",
]
LABEL = "state"

STATE_OPTIONS = [
    "NORMAL",
    "OVERHEATING",
    "HIGH_VIBRATION",
    "LOW_PRESSURE",
    "MOTOR_OVERLOAD",
]
STATE_WEIGHTS = [60, 10, 10, 10, 10]


def generate_sensor_data(rng):
    """Generate one labeled sensor reading."""
    temperature = round(rng.uniform(55, 72), 2)
    pressure = round(rng.uniform(4.0, 6.5), 2)
    vibration = round(rng.uniform(1.0, 3.5), 2)
    current = round(rng.uniform(8.0, 16.0), 2)
    rpm = rng.randint(1400, 2200)

    state = rng.choices(STATE_OPTIONS, weights=STATE_WEIGHTS, k=1)[0]

    if state == "OVERHEATING":
        temperature = round(rng.uniform(76, 98), 2)

        if rng.random() < 0.15:
            temperature = round(rng.uniform(70, 79), 2)

    elif state == "HIGH_VIBRATION":
        vibration = round(rng.uniform(5.0, 10.0), 2)
        if rng.random() < 0.15:
            vibration = round(rng.uniform(3.0, 5.8), 2)

    elif state == "LOW_PRESSURE":
        pressure = round(rng.uniform(1.0, 3.5), 2)
        if rng.random() < 0.15:
            pressure = round(rng.uniform(3.0, 4.3), 2)

    elif state == "MOTOR_OVERLOAD":
        current = round(rng.uniform(19.0, 29.0), 2)
        if rng.random() < 0.15:
            current = round(rng.uniform(15.0, 20.5), 2)

    return {
        "temperature": temperature,
        "pressure": pressure,
        "vibration": vibration,
        "current": current,
        "rpm": rpm,
        "state": state,
    }


def generate_dataset(samples, output_path, seed=42):
    if samples < len(STATE_OPTIONS):
        raise ValueError(
            f"Use at least {len(STATE_OPTIONS)} samples so every state can be represented."
        )

    rng = random.Random(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [generate_sensor_data(rng) for _ in range(samples)]


    for i, state in enumerate(STATE_OPTIONS):
        rows[i]["state"] = state

        row = rows[i]
        if state == "OVERHEATING":
            row["temperature"] = round(rng.uniform(76, 98), 2)
        elif state == "HIGH_VIBRATION":
            row["vibration"] = round(rng.uniform(5, 10), 2)
        elif state == "LOW_PRESSURE":
            row["pressure"] = round(rng.uniform(1, 3.5), 2)
        elif state == "MOTOR_OVERLOAD":
            row["current"] = round(rng.uniform(19, 29), 2)

    rng.shuffle(rows)

    fieldnames = FEATURES + [LABEL]
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    counts = Counter(row[LABEL] for row in rows)
    print(f"Generated {len(rows):,} rows: {output_path.resolve()}")
    print("\nClass distribution:")
    for state in STATE_OPTIONS:
        count = counts[state]
        print(f"  {state:16s} {count:6,d} ({count / len(rows) * 100:5.1f}%)")
    print("\nFeatures:", ", ".join(FEATURES))
    print("Target label:", LABEL)
    print("Note: This is synthetic data; validate a trained model on real sensor data before relying on it.")


def main():
    parser = argparse.ArgumentParser(description="Generate Day 2 machine sensor training data.")
    parser.add_argument("--samples", type=int, default=250, help="Number of rows to generate (default: 10000)")
    parser.add_argument("--output", default="day2_sample.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible data")
    args = parser.parse_args()

    generate_dataset(args.samples, args.output, args.seed)


if __name__ == "__main__":
    main()
