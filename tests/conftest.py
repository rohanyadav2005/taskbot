import json
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.json"))


@pytest.fixture
def spec_dict():
    """A fresh, mutable copy of the 2-DOF example to break in tests."""
    return json.loads((EXAMPLES_DIR / "arm_2dof.json").read_text())
