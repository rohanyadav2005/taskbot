import pytest
from pydantic import ValidationError

from taskbot import CheckResult


def test_round_trips_to_required_dict_shape():
    r = CheckResult(
        check="joint_torque",
        passed=False,
        value=12.4,
        limit=8.0,
        message="shoulder_pitch needs 12.4 N·m but the motor gives 8.0 N·m.",
    )
    assert r.model_dump() == {
        "check": "joint_torque",
        "passed": False,
        "value": 12.4,
        "limit": 8.0,
        "message": "shoulder_pitch needs 12.4 N·m but the motor gives 8.0 N·m.",
    }


def test_value_and_limit_optional_for_non_numeric_checks():
    r = CheckResult(check="self_collision", passed=True, message="No self-collisions.")
    assert r.value is None and r.limit is None


@pytest.mark.parametrize("field", ["check", "message"])
def test_blank_text_rejected(field):
    kwargs = dict(check="reach", passed=True, message="ok")
    kwargs[field] = "   "
    with pytest.raises(ValidationError):
        CheckResult(**kwargs)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_numbers_rejected(bad):
    with pytest.raises(ValidationError):
        CheckResult(check="reach", passed=False, value=bad, message="x")


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        CheckResult(check="reach", passed=True, message="ok", units="m")


def test_frozen():
    r = CheckResult(check="reach", passed=True, message="ok")
    with pytest.raises(ValidationError):
        r.passed = False
