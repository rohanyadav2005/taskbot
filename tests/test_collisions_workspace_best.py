"""Task 6.5: verified self-collision fixes, workspaces that contain the base, and keeping the
best design. Built around the fourth real-model run: the shoulder 0.03 m above a 0.15 m base,
a workspace "a 0.3 m cube around the base", and a worse final round."""

import json
import re

import pytest
from conftest import example_dict, strong_arm, with_motor
from pydantic import ValidationError

from taskbot import checks
from taskbot.build_urdf import build_model
from taskbot.checks import CheckResult
from taskbot.llm import FakeClient
from taskbot.repair import Round, best_round, repair_loop, round_rank
from taskbot.report import render_report, write_outputs
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec, load_motor_catalog

CATALOG = load_motor_catalog()

# Run 4's requirements, with the workspace moved in front of the robot.
RUN4 = dict(task_summary="water the three plants on my windowsill", robot_type="arm", reach_m=0.3,
            payload_kg=0.3, workspace={"x_min": 0.12, "x_max": 0.25, "y_min": -0.12, "y_max": 0.12,
                                       "z_min": 0.0, "z_max": 0.25},
            dof_min=3, gripper_needed=True, gripper_kind="parallel", position_tolerance_m=0.01,
            tolerance_reason="1 cm is enough to water a pot", assumptions=[])


def low_shoulder() -> dict:
    """The shoulder 0.03 m above a 0.15 x 0.15 m base, with a wide shoulder range."""
    d = example_dict("arm_4dof")
    d["modules"][1]["segments"][0]["link"]["length_m"] = 0.03
    d["modules"][1]["segments"][1]["joint"] |= {"lower": -2.0, "upper": 2.0}
    return d


def collide(d: dict, req=RUN4):
    spec = RobotSpec.model_validate(d)
    (r,) = checks.self_collision(spec, Requirements(**req), build_model(spec))
    return r


def apply_fix(d: dict, fix: str) -> dict:
    """Apply one suggestion as the model would, from its text alone."""
    d = json.loads(json.dumps(d))
    segs = d["modules"][1]["segments"]
    if m := re.match(r"set (\w+) length_m to at least ([\d.]+) m", fix):
        next(s for s in segs if s["link"]["name"] == m[1])["link"]["length_m"] = float(m[2])
    elif m := re.match(r"set (\w+) (lower|upper) to (-?[\d.]+) rad", fix):
        next(s for s in segs if s["joint"]["name"] == m[1])["joint"][m[2]] = float(m[3])
    elif m := re.match(r"make the base footprint ([\d.]+) x ([\d.]+) m", fix):
        d["modules"][0]["footprint_m"] = [float(m[1]), float(m[2])]
    else:
        raise AssertionError(f"unparsed fix: {fix}")
    return d


def fixes_in(message: str) -> list[str]:
    tail = message.split("smallest change first: ", 1)[1].rstrip(".")
    return [re.sub(r" \(collisions [\d.]+%\)$", "", f) for f in tail.split("; or ")]


# --------------------------------------------------------------------------- self-collision


def test_low_shoulder_gets_exact_verified_fixes():
    r = collide(low_shoulder())
    assert not r.passed
    assert "shoulder_pitch sits only 0.03 m above the base top, and the base is 0.15 x 0.15 m" in r.message
    assert "Verified fixes, each re-checked with the same sampling, smallest change first: " in r.message
    assert re.search(r"set turret length_m to at least [\d.]+ m \(collisions [\d.]+%\)", r.message)


def test_every_suggested_fix_actually_passes():
    d = low_shoulder()
    fixes = fixes_in(collide(d).message)
    assert fixes
    for fix in fixes:
        assert collide(apply_fix(d, fix)).passed, fix


def test_oversized_base_fix_is_verified_and_listed_first():
    """The standard 0.05 m turret on a 0.4 m base: the upper arm hits the plate's edge. Shrinking
    the base is the smaller change, so it comes first; every listed fix passes when applied."""
    d = example_dict("arm_4dof")
    d["modules"][0]["footprint_m"] = [0.4, 0.4]
    fixes = fixes_in(collide(d).message)
    assert fixes[0].startswith("make the base footprint") and len(fixes) >= 2
    for fix in fixes:
        assert collide(apply_fix(d, fix)).passed, fix


def test_no_verified_fix_is_said_plainly(monkeypatch):
    """With the riser search disabled, nothing else fixes the low shoulder: no fix is offered."""
    monkeypatch.setattr(checks, "RISER_MAX_EXTRA_M", 0.0)
    r = collide(low_shoulder())
    assert not r.passed and "Verified fixes" not in r.message
    assert ("No single change to the riser length, the first pitch joint's range or the base size brought "
            "collisions under 5% in the same sampling.") in r.message


# --------------------------------------------------------------------------- workspace vs base


def test_workspace_around_the_base_is_rejected():
    cube = RUN4 | {"workspace": {"x_min": -0.15, "x_max": 0.15, "y_min": -0.15, "y_max": 0.15,
                                 "z_min": 0.0, "z_max": 0.3}}
    with pytest.raises(ValidationError) as err:
        Requirements(**cube)
    msg = str(err.value)
    assert "includes the robot's own base (|x| and |y| under 0.08 m, z under 0.1 m)" in msg
    assert "plants in a row in front of the robot are at x from about 0.15 to 0.35 m" in msg


@pytest.mark.parametrize("ws", [
    {"x_min": 0.12, "x_max": 0.25, "y_min": -0.12, "y_max": 0.12, "z_min": 0.0, "z_max": 0.25},   # in front
    {"x_min": -0.1, "x_max": 0.1, "y_min": -0.1, "y_max": 0.1, "z_min": 0.12, "z_max": 0.25},    # above it
])
def test_workspace_clear_of_the_base_is_accepted(ws):
    Requirements(**(RUN4 | {"workspace": ws}))


def test_json_retry_moves_the_workspace_off_the_base():
    cube = RUN4 | {"workspace": {"x_min": -0.15, "x_max": 0.15, "y_min": -0.15, "y_max": 0.15,
                                 "z_min": 0.0, "z_max": 0.3}}
    fake = FakeClient({"Requirements": [cube, RUN4]})
    assert fake.complete_json("sys", "water", Requirements).workspace.x_min == 0.12
    assert "includes the robot's own base" in fake.calls[1][1][-1]["content"]


# --------------------------------------------------------------------------- best round


def fails(*checks_: tuple[str, float, float]) -> list[CheckResult]:
    return [CheckResult(check=c, passed=False, value=v, limit=lim, message="m") for c, v, lim in checks_]


def test_rounds_rank_by_failures_then_shortfall():
    spec = RobotSpec.model_validate(strong_arm())
    two = Round(1, spec, fails(("reach", 0.03, 0.01), ("self_collision", 0.7, 0.05)))
    four = Round(2, spec, fails(("reach", 0.02, 0.01), ("static_torque", 2, 1), ("mass", 1, 1),
                                ("joint_limits", 1.5, 1.57)))
    two_smaller = Round(3, spec, fails(("reach", 0.02, 0.01), ("self_collision", 0.06, 0.05)))
    assert round_rank(two)[0] == 2 and round_rank(four)[0] == 4
    assert best_round([two, four]) is two
    assert best_round([two, four, two_smaller]) is two_smaller
    tie = Round(4, spec, two_smaller.results)
    assert best_round([two_smaller, tie]) is tie  # the later round on a tie


def test_worse_round_restarts_from_the_best_design_which_is_kept(desk_req, tmp_path):
    weak = with_motor(strong_arm(), "shoulder_pitch", "mg996r")          # round 1: torque fails
    worse = with_motor(example_dict("arm_2dof"), "shoulder_pitch", "mg996r")  # round 2: more fails
    other = with_motor(strong_arm(), "shoulder_pitch", "sts3215")        # round 3: anything
    fake = FakeClient({"RobotSpec": [worse, other]})
    events = []
    outcome = repair_loop(fake, desk_req, CATALOG, initial_spec=RobotSpec.model_validate(weak),
                          max_rounds=3, simulate=False, on_event=events.append)
    r1, r2, _ = outcome.rounds
    assert round_rank(r2) > round_rank(r1)
    note = (f"Round 2 was worse than round 1 ({len({f.check for f in r2.failures})} failures vs "
            f"{len({f.check for f in r1.failures})}). Starting again from round 1's design.")
    assert any(note in e for e in events)
    third_request = fake.calls[1][1][-1]["content"]
    assert third_request.startswith("Your robot design failed engineering checks. Fix it.\nFailed checks:\n" + note)
    current = json.loads(third_request.split("Current spec: ", 1)[1].split("\nRequirements: ", 1)[0])
    assert current == json.loads(r1.spec.model_dump_json(exclude_none=True))  # round 1's design
    # The final design is the best round, wherever it came.
    best = best_round(outcome.rounds)
    assert outcome.final is best and outcome.last.number == 3
    report = render_report(outcome)
    if best.number != 3:
        assert f"The best design, from round {best.number}" in report
    out = write_outputs(outcome, tmp_path, CATALOG, record=False)
    saved = RobotSpec.model_validate_json((out / "spec.json").read_text(encoding="utf-8"))
    assert saved.model_dump() == best.spec.model_dump()


def test_last_round_reported_as_last_when_it_is_the_best(desk_req):
    weak = with_motor(strong_arm(), "shoulder_pitch", "mg996r")
    outcome = repair_loop(FakeClient({"RobotSpec": []}), desk_req, CATALOG,
                          initial_spec=RobotSpec.model_validate(weak), max_rounds=1, simulate=False)
    assert "The last design still fails" in render_report(outcome)
