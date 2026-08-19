import json
from pathlib import Path


def test_maintained_mechint_notebooks_are_valid_notebook_json():
    repo_root = Path(__file__).resolve().parents[3]
    tutorial_root = repo_root / "tutorials" / "mechint"
    notebooks = sorted(tutorial_root.glob("*.ipynb"))

    assert notebooks
    assert (tutorial_root / "05_shared_neural_computation_study.ipynb") in notebooks

    for path in notebooks:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["nbformat"] == 4, path
        assert payload["cells"], path
        assert all("cell_type" in cell and "source" in cell for cell in payload["cells"]), path
