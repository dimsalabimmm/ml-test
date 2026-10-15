"""Check notebook syntax and exported metrics without rerunning training."""

import ast
import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def metrics(rows, column):
    truth = [int(row["true_label"]) for row in rows]
    predictions = [int(row[column]) for row in rows]
    accuracy = sum(a == b for a, b in zip(truth, predictions)) / len(rows)
    f1_values = []
    for label in [0, 1, 2]:
        tp = sum(a == label and b == label for a, b in zip(truth, predictions))
        fp = sum(a != label and b == label for a, b in zip(truth, predictions))
        fn = sum(a == label and b != label for a, b in zip(truth, predictions))
        f1_values.append(2 * tp / (2 * tp + fp + fn))
    return accuracy, sum(f1_values) / 3


def main():
    notebook = json.loads((ROOT / "ml_test.ipynb").read_text(encoding="utf-8"))
    for number, cell in enumerate(notebook["cells"], 1):
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]), filename=f"cell-{number}")
            assert not any(
                output.get("output_type") == "error"
                for output in cell.get("outputs", [])
            ), f"Saved error in cell {number}"

    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "\x1b" not in requirements, "ANSI escape in requirements"

    with (ROOT / "artifacts/test_predictions.csv").open(encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 20_074
    assert [int(row["test_row"]) for row in rows] == list(range(len(rows)))
    with (ROOT / "artifacts/test_metrics.csv").open(encoding="utf-8") as file:
        exported = list(csv.DictReader(file))
    for result, column in zip(exported, ["baseline_prediction", "neural_prediction"]):
        accuracy, macro_f1 = metrics(rows, column)
        assert math.isclose(accuracy, float(result["Test accuracy"]), abs_tol=1e-12)
        assert math.isclose(macro_f1, float(result["Test Macro-F1"]), abs_tol=1e-12)
    print("OK: notebook syntax, saved outputs, requirements and test metrics")


if __name__ == "__main__":
    main()
