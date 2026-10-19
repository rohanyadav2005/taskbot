import pytest
from pydantic import ValidationError

from conftest import EXAMPLES
from taskbot.schema import Joint, Link, Motor, RobotSpec, load_motor_catalog


def arm(d):
    return d["modules"][1]["segments"]


# --------------------------------------------------------------------------- good specs


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_examples_validate(path):
    spec = RobotSpec.model_validate_json(path.read_text(encoding="utf-8"))
    assert spec.modules[0].kind == "fixed_base"
    assert spec.modules[-1].kind == "gripper"


def test_examples_cover_2_4_6_dof():
    dofs = {}
    for p in EXAMPLES:
        spec = RobotSpec.model_validate_json(p.read_text(encoding="utf-8"))
        dofs[p.stem] = sum(len(m.segments) for m in spec.modules if m.kind == "arm_chain")
    assert dofs == {"arm_2dof": 2, "arm_4dof": 4, "arm_6dof": 6}


def test_catalog_loads_with_sources_for_every_value():
    catalog = load_motor_catalog()
    assert 6 <= len(catalog) <= 12
    numeric = {"stall_torque_nm", "rated_torque_nm", "max_speed_rad_s", "no_load_speed_rad_s",
               "mass_kg", "body_size_m", "price_usd"}
    for m in catalog.values():
        needed = numeric | ({"dynamixel_gains"} if m.kind == "dynamixel" else set()) \
            | ({"pull_out_curve"} if m.kind == "stepper" else set())  # says why there is none
        assert needed <= set(m.sources), f"{m.id} missing sources for {needed - set(m.sources)}"


def test_motor_ids_can_come_from_context(spec_dict):
    with pytest.raises(ValidationError, match="unknown motor_id 'sts3215'"):
        RobotSpec.model_validate(spec_dict, context={"motor_ids": {"mg996r"}})


# --------------------------------------------------------------------------- bad specs
# Each case mutates the 2-DOF example and checks that the error message names
# the actual problem, since these messages go back to the LLM.


def _dup_link(d):
    arm(d)[1]["link"]["name"] = "turret"


def _dup_joint_vs_link(d):
    arm(d)[1]["joint"]["name"] = "turret"


def _collides_with_generated_name(d):
    arm(d)[1]["link"]["name"] = "gripper_palm"


def _base_not_first(d):
    d["modules"].reverse()


def _two_bases(d):
    d["modules"].insert(1, dict(d["modules"][0], name="base2"))


def _gripper_not_last(d):
    d["modules"].insert(1, d["modules"].pop())


def _no_arm(d):
    del d["modules"][1]


def _unknown_motor(d):
    arm(d)[0]["joint"]["motor_id"] = "warp_drive"


def _unknown_kind(d):
    d["modules"][1]["kind"] = "legs"


def _extra_field(d):
    arm(d)[0]["link"]["mass_kg"] = 0.1


def _bad_name(d):
    arm(d)[0]["joint"]["name"] = "base yaw"


def _negative_size(d):
    arm(d)[1]["link"]["size_m"] = [0.025, -0.025, 0.12]


BAD = {
    "duplicate link name": (_dup_link, "name 'turret' is used by both"),
    "joint/link name clash": (_dup_joint_vs_link, "name 'turret' is used by both"),
    "clash with generated gripper name": (_collides_with_generated_name, "'gripper_palm'"),
    "base not first": (_base_not_first, "modules\\[0\\] must be a fixed_base"),
    "two bases": (_two_bases, "only one base \\(fixed_base or wheeled_base\\)"),
    "gripper not last": (_gripper_not_last, "must be the last module"),
    "no arm": (_no_arm, "at least one arm_chain"),
    "unknown motor": (_unknown_motor, "unknown motor_id 'warp_drive'"),
    "unknown module kind": (_unknown_kind, "legs"),
    "mass given": (_extra_field, "mass_kg"),
    "name with space": (_bad_name, "pattern"),
    "negative size": (_negative_size, "greater than 0"),
}


@pytest.mark.parametrize("mutate,match", BAD.values(), ids=BAD.keys())
def test_bad_specs_rejected_with_specific_message(spec_dict, mutate, match):
    mutate(spec_dict)
    with pytest.raises(ValidationError, match=match):
        RobotSpec.model_validate(spec_dict)


JOINT = dict(name="j", type="revolute", axis="z", lower=-1.0, upper=1.0, motor_id="sts3215")


@pytest.mark.parametrize(
    "changes,match",
    [
        (dict(lower=1.0, upper=-1.0), "lower .* must be less than upper"),
        (dict(lower=None), "needs both lower and upper"),
        (dict(type="continuous"), "continuous, so it must not set lower/upper"),
        (dict(type="prismatic", upper=None), "needs both lower and upper"),
        (dict(axis="w"), "axis"),
    ],
)
def test_bad_joints(changes, match):
    with pytest.raises(ValidationError, match=match):
        Joint(**(JOINT | changes))


def test_continuous_joint_without_limits_ok():
    Joint(**(JOINT | dict(type="continuous", lower=None, upper=None)))


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(shape="box", radius_m=0.1, length_m=0.1), "is a box: give size_m"),
        (dict(shape="box", size_m=(0.1, 0.1, 0.1), radius_m=0.1), "is a box"),
        (dict(shape="cylinder", radius_m=0.1), "is a cylinder: give radius_m and length_m"),
        (dict(shape="cylinder", size_m=(0.1, 0.1, 0.1)), "is a cylinder"),
        (dict(shape="box", size_m=(0.1, 0.1, 0.1), material="wood"), "material"),
    ],
)
def test_bad_links(kwargs, match):
    with pytest.raises(ValidationError, match=match):
        Link(**({"name": "l", "material": "pla"} | kwargs))


def test_parallel_gripper_needs_motor(spec_dict):
    del spec_dict["modules"][-1]["motor_id"]
    with pytest.raises(ValidationError, match="parallel and needs a motor_id"):
        RobotSpec.model_validate(spec_dict)


def test_rated_torque_above_stall_rejected():
    m = load_motor_catalog()["sts3215"].model_dump()
    m["rated_torque_nm"] = m["stall_torque_nm"] * 2
    with pytest.raises(ValidationError, match="exceeds stall_torque_nm"):
        Motor(**m)


def test_steppers_are_unsupported_with_a_reason():
    from taskbot.schema import supported_motors

    catalog = load_motor_catalog()
    nema = catalog["nema17_pg27"]
    assert not nema.supported and "skip steps" in nema.unsupported_reason
    assert "nema17_pg27" not in supported_motors(catalog)
    assert all(m.supported for m in catalog.values() if m.kind != "stepper")


def test_unsupported_motor_needs_a_reason():
    data = load_motor_catalog()["sts3215"].model_dump() | {"supported": False}
    with pytest.raises(ValidationError, match="unsupported_reason"):
        Motor(**data)


def test_design_step_rejects_unsupported_motors(desk_req):
    import json

    from conftest import example_dict, with_motor
    from taskbot.design import design_robot
    from taskbot.llm import FakeClient, LLMOutputError

    nema = with_motor(example_dict("arm_4dof"), "shoulder_pitch", "nema17_pg27")
    fake = FakeClient({"RobotSpec": [nema] * 4})
    with pytest.raises(LLMOutputError, match="unknown motor_id 'nema17_pg27'"):
        design_robot(fake, desk_req, load_motor_catalog())
    assert json.dumps(nema)  # the same spec still loads outside the design step
    RobotSpec.model_validate(nema)
