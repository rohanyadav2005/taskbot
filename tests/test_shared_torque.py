"""static_torque and torque_at_speed are thin wrappers over holding_loads and speed_loads,
the functions the sizer uses: one implementation of the torque math."""

import pytest
from conftest import DESK_REQ, strong_arm, with_motor

from taskbot import checks
from taskbot.build_urdf import build_model
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec


@pytest.mark.parametrize("motor", ["xm540_w270", "mg996r"])
def test_static_torque_values_are_holding_loads(motor):
    spec = RobotSpec.model_validate(with_motor(strong_arm(), "shoulder_pitch", motor))
    req, model = Requirements(**DESK_REQ), build_model(spec)
    loads = checks.holding_loads(spec, req, model)
    results = checks.static_torque(spec, req, model)
    assert [r.value for r in results] == [jl.need for jl in loads]
    assert [r.limit for r in results] == [checks.usable_torque(checks._catalog(None)[jl.joint.motor_id])
                                          for jl in loads]


def test_torque_at_speed_reports_the_worst_speed_load():
    spec = RobotSpec.model_validate(with_motor(strong_arm(), "shoulder_pitch", "mg996r"))
    req = Requirements(**DESK_REQ)
    loads = checks.speed_loads(spec, req)
    worst = max(loads, key=lambda load: load.ratio)
    (r,) = checks.torque_at_speed(spec, req, build_model(spec))
    assert (r.value, r.limit) == (worst.gravity, worst.available)
    assert r.message.startswith(worst.joint.name)
