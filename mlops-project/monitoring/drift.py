"""Drift detection pipeline using Evidently."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.metric_preset import DataDriftPreset

from src.utils import ensure_dir, load_dataset


def generate_drift_report(output_dir: str = "reports/drift") -> Path:
    x_train, x_test, _, _ = load_dataset()

    # Simulate current production data by perturbing one feature.
    current_data = x_test.copy()
    current_data["mean radius"] = current_data["mean radius"] * 1.15

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=x_train, current_data=current_data)

    output_path = ensure_dir(output_dir) / "drift_report.html"
    report.save_html(str(output_path))
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Evidently drift report")
    parser.add_argument("--output-dir", default="reports/drift")
    args = parser.parse_args()

    report_path = generate_drift_report(output_dir=args.output_dir)
    print(f"Drift report saved to {report_path}")


if __name__ == "__main__":
    main()
