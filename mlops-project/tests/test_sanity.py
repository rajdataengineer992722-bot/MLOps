from pathlib import Path


def test_project_structure_exists():
    root = Path(__file__).resolve().parents[1]
    assert (root / "src" / "train.py").exists()
    assert (root / "api" / "app.py").exists()
    assert (root / "monitoring" / "drift.py").exists()
