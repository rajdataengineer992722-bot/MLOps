"""Data drift detection workflow powered by Evidently."""

from __future__ import annotations

import argparse

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from src.utils import DATA_DIR, ensure_dir, load_dataset, load_environment, save_json, save_monitoring_datasets


def load_monitoring_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load reference and current monitoring datasets, generating them if needed."""
    reference_path = DATA_DIR / "monitoring" / "reference.csv"
    current_path = DATA_DIR / "monitoring" / "current.csv"
    if reference_path.exists() and current_path.exists():
        return pd.read_csv(reference_path), pd.read_csv(current_path)

    x_train, x_test, _, _ = load_dataset()
    current_data = x_test.copy()
    current_data["monthly_charges"] = (current_data["monthly_charges"] * 1.12).round(2)
    current_data["contract_type"] = current_data["contract_type"].replace({"two-year": "month-to-month"})
    save_monitoring_datasets(x_train, current_data)
    return x_train, current_data


def generate_drift_report(output_dir: str = "reports/drift") -> dict[str, str]:
    """Generate and persist an Evidently drift report."""
    load_environment()
    reference_data, current_data = load_monitoring_frames()

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    destination = ensure_dir(output_dir)
    html_path = destination / "drift_report.html"
    json_path = destination / "drift_report.json"
    report.save_html(str(html_path))
    save_json(report.as_dict(), json_path)
    return {"html_report": str(html_path), "json_report": str(json_path)}


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Generate an Evidently drift report")
    parser.add_argument("--output-dir", default="reports/drift")
    args = parser.parse_args()

    outputs = generate_drift_report(output_dir=args.output_dir)
    print(f"Drift HTML report: {outputs['html_report']}")
    print(f"Drift JSON report: {outputs['json_report']}")


if __name__ == "__main__":
    main()
