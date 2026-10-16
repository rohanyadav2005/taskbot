"""Wheeled bases and mobile manipulators: schema, URDF, static checks, driving sims,
the fake mobile scenarios, the out-of-scope stop, the GLB export and the app's run loop."""


import mujoco
import pytest
from pydantic import ValidationError

from taskbot.build_urdf import TERRAIN, build_mjspec, build_model, build_urdf
from taskbot.checks import run_checks
from taskbot.fake import _load, fake_client, scenario
from taskbot.mobile_checks import MOBILE_CHECKS, tip_over, top_speed, traction, wheel_torque
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec, load_motor_catalog, supported_motors
from taskbot.sim.motor import motor_model

CATALOG = load_motor_catalog()


def spec_of(d: dict) -> RobotSpec:
    return RobotSpec.model_validate(d, context={"catalog": CATALOG})


def req_of(name: str, **changes) -> Requirements:
    return Requirements(**(_load(f"{name}_requirements") | changes))


def design(name: str) -> dict:
    return _load(name)


def check(fn, spec: RobotSpec, req: Requirements):
    (result,) = fn(spec, req, build_model(spec, CATALOG), CATALOG)
    return result


# --------------------------------------------------------------------------- schema


def test_wheel_motor_must_support_velocity_mode():
    d = design("mobile_design_round2")
    d["modules"][0]["wheel_motor_id"] = "mg996r"
    with pytest.raises(ValidationError, match="needs a motor with velocity control.*mg996r"):
        spec_of(d)


def test_arm_joint_must_support_position_mode():
    d = design("mobile_manipulator_design_round2")
    d["modules"][1]["segments"][0]["joint"]["motor_id"] = "pololu_37d_50"
    with pytest.raises(ValidationError, match="needs a motor with position control.*pololu_37d_50"):
        spec_of(d)


def test_caster_only_on_differential_drive():
    d = design("mobile_design_round2")
    d["modules"][0]["drive"] = "four_wheel"
    with pytest.raises(ValidationError, match="caster"):
        spec_of(d)


def test_wheeled_base_must_come_first():
    d = design("mobile_manipulator_design_round2")
    d["modules"] = [d["modules"][1], d["modules"][0], d["modules"][2]]
    with pytest.raises(ValidationError):
        spec_of(d)


def test_battery_defaults_to_an_estimate_recorded_as_an_assumption():
    base = spec_of(design("mobile_design_round2")).base
    assert base.battery_mass_kg is None
    assert base.battery_kg == pytest.approx(0.25)
    assert "estimated" in base.battery_note


def test_mobile_requirements_need_the_mobile_fields():
    d = _load("mobile_requirements")
    del d["speed_m_s"]
    with pytest.raises(ValidationError, match="speed_m_s"):
        Requirements(**d)


def test_wheel_catalog_has_sourced_velocity_motors():
    wheel = {k: m for k, m in supported_motors(CATALOG).items() if "velocity" in m.modes}
    assert len(wheel) >= 4
    assert {"pololu_37d_50", "pololu_25d_34_hp", "jgb37_520_110", "xl430_w250"} <= set(wheel)
    for m in wheel.values():
        assert m.sources and m.voltage_v and m.price_usd > 0 and m.mass_kg > 0


# --------------------------------------------------------------------------- velocity presets


def test_dc_velocity_preset_is_the_documented_one():
    m = CATALOG["pololu_37d_50"]
    mm = motor_model(m, mode="velocity")
    assert mm.kp == pytest.approx(m.stall_torque_nm / (0.25 * m.no_load_speed_rad_s))
    assert mm.ki == pytest.approx(mm.kp / 0.2)
    assert "25% speed error" in mm.gain_source


def test_dynamixel_velocity_preset_uses_factory_gains():
    mm = motor_model(CATALOG["xm430_w350"], mode="velocity")
    assert "factory velocity gains" in mm.gain_source and "unverified" in mm.gain_source
    # Same torque-speed limit as position mode.
    pos = motor_model(CATALOG["xm430_w350"])
    assert mm.drive_limit(3.0) == pytest.approx(pos.drive_limit(3.0))


def test_velocity_mode_refused_for_position_only_motor():
    with pytest.raises(ValueError, match="velocity"):
        motor_model(CATALOG["mg996r"], mode="velocity")


# --------------------------------------------------------------------------- URDF / MJCF


def test_wheels_are_continuous_joints_and_caster_is_a_ball():
    spec = spec_of(design("mobile_design_round2"))
    urdf = build_urdf(spec, CATALOG)
    assert urdf.count('type="continuous"') == 2
    model = build_mjspec(spec, CATALOG, floating=True, terrain="carpet").compile()
    types = [model.jnt_type[i] for i in range(model.njnt)]
    assert types.count(mujoco.mjtJoint.mjJNT_FREE) == 1
    assert types.count(mujoco.mjtJoint.mjJNT_BALL) == 1
    names = {model.actuator(i).name for i in range(model.nu)}
    assert names == {"base_wheel_left_joint_motor", "base_wheel_right_joint_motor"}


def test_wheel_friction_comes_from_the_terrain_table():
    spec = spec_of(design("mobile_design_round2"))
    for terrain, values in TERRAIN.items():
        model = build_mjspec(spec, CATALOG, floating=True, terrain=terrain).compile()
        wheel = model.body("base_wheel_left").id
        geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == wheel and model.geom_contype[g]]
        assert geoms and all(model.geom_friction[g][0] == pytest.approx(values["mu"]) for g in geoms)


def test_check_model_keeps_the_base_fixed():
    model = build_model(spec_of(design("mobile_design_round2")), CATALOG)
    assert mujoco.mjtJoint.mjJNT_FREE not in [model.jnt_type[i] for i in range(model.njnt)]


# --------------------------------------------------------------------------- static checks


@pytest.mark.parametrize("name", ["mobile", "mobile_manipulator"])
def test_round_two_reference_designs_pass_the_mobile_checks(name):
    spec, req = spec_of(design(f"{name}_design_round2")), req_of(name)
    model = build_model(spec, CATALOG)
    results = [r for fn in MOBILE_CHECKS for r in fn(spec, req, model, CATALOG)]
    assert all(r.passed for r in results), [r.message for r in results if not r.passed]


def test_wheel_torque_fails_speed_limited_when_short_only_at_speed():
    r = check(wheel_torque, spec_of(design("mobile_design_round1")), req_of("mobile"))
    assert not r.passed and r.cause == "speed_limited"
    assert "jgb37_520_110" in r.message and "faster wheel motor" in r.message
    assert r.value > r.limit


def test_wheel_torque_fails_saturated_when_short_even_at_rest():
    r = check(wheel_torque, spec_of(design("mobile_design_round1")), req_of("mobile", base_payload_kg=25.0))
    assert not r.passed and r.cause == "saturated"
    assert "stronger wheel motor" in r.message and "four_wheel" in r.message


def test_top_speed_fails_with_a_fix():
    r = check(top_speed, spec_of(design("mobile_design_round1")), req_of("mobile"))
    assert not r.passed and r.cause == "speed_limited"
    assert r.value == pytest.approx(0.8 * CATALOG["jgb37_520_110"].no_load_speed_rad_s * 0.05)
    assert "faster wheel motor or larger wheels" in r.message


def test_tip_over_fails_with_arm_reaching_past_the_wheels():
    r = check(tip_over, spec_of(design("mobile_manipulator_design_round1")), req_of("mobile_manipulator"))
    assert not r.passed and r.cause == "unstable"
    assert "arm extended towards 0°" in r.message and "outside the support polygon" in r.message
    for fix in ("wider track", "longer wheelbase", "arm mount", "shorter arm reach"):
        assert fix in r.message


def test_tip_over_fails_on_two_wheels_without_a_caster():
    d = design("mobile_design_round2")
    d["modules"][0]["caster"] = False
    r = check(tip_over, spec_of(d), req_of("mobile"))
    assert not r.passed and r.cause == "unstable" and "Add a caster" in r.message


def test_traction_fails_on_a_steep_slippery_slope():
    d = design("mobile_design_round2")
    d["modules"][0] |= {"drive": "four_wheel", "caster": False}
    r = check(traction, spec_of(d), req_of("mobile", max_incline_deg=30.0, terrain="outdoor"))
    assert not r.passed and r.cause == "slip"
    assert "μ=0.5 on outdoor" in r.message and "slip" in r.message
    assert r.value > r.limit


def test_run_checks_picks_the_checks_for_the_robot_type():
    spec, req = spec_of(design("mobile_design_round2")), req_of("mobile")
    names = {r.check for r in run_checks(spec, req, CATALOG)}
    assert {"wheel_torque", "top_speed", "tip_over", "traction", "budget", "mass"} <= names
    assert not names & {"reach", "static_torque", "workspace_corners"}


def test_wrong_robot_type_is_a_failure():
    spec = spec_of(design("mobile_design_round2"))
    results = run_checks(spec, req_of("mobile_manipulator"), CATALOG)
    wrong = [r for r in results if r.cause == "wrong_type"]
    assert wrong and "mobile_manipulator" in wrong[0].message


# --------------------------------------------------------------------------- driving sims


@pytest.fixture(scope="module")
def sims():
    from taskbot import mobile_sim as ms

    mobile = spec_of(design("mobile_design_round2")), req_of("mobile")
    mm = spec_of(design("mobile_manipulator_design_round2")), req_of("mobile_manipulator")
    runs = {f"mobile/{r.name}": r for r in ms.run_mobile_sim_tests(*mobile, CATALOG)}
    runs |= {f"mm/{r.name}": r for r in ms.run_mobile_sim_tests(*mm, CATALOG)}
    return runs


def test_mobile_reference_design_passes_the_driving_sims(sims):
    for name in ("drive_straight", "turn_in_place", "incline"):
        r = sims[f"mobile/{name}"].result
        assert r.passed, r.message
    assert "mobile/drive_with_arm_extended" not in sims


def test_mobile_manipulator_passes_drive_with_arm_extended(sims):
    r = sims["mm/drive_with_arm_extended"].result
    assert r.passed, r.message
    for side in ("front", "left", "right"):  # "back" is outside base_yaw's ±1.6 rad, so skipped
        assert f"arm {side}" in r.message


def test_drive_straight_reports_distance_heading_and_slip(sims):
    msg = sims["mobile/drive_straight"].result.message
    assert "actually moved" in msg and "heading error" in msg and "wheel slip" in msg


def test_turn_in_place_reaches_90_degrees(sims):
    r = sims["mobile/turn_in_place"].result
    assert r.value < 5.0 and "ended at" in r.message


def test_overloaded_base_fails_drive_straight_and_incline_with_wheel_advice():
    from taskbot.mobile_sim import drive_straight, incline

    spec, req = spec_of(design("mobile_design_round1")), req_of("mobile", base_payload_kg=25.0)
    for test in (drive_straight, incline):
        r = test(spec, req, CATALOG).result
        assert not r.passed and r.cause == "saturated", r.message
        assert "jgb37_520_110" in r.message and "stronger wheel motor" in r.message
        assert "outer links" not in r.message  # the arm advice doesn't apply to a wheel


def test_incline_failure_says_it_slid_back():
    from taskbot.mobile_sim import incline

    spec = spec_of(design("mobile_design_round1"))
    r = incline(spec, req_of("mobile", max_incline_deg=20.0, base_payload_kg=8.0), CATALOG).result
    assert not r.passed and "backwards down the slope" in r.message and r.value > 0.01


def test_narrow_base_tips_with_arm_extended():
    from taskbot.mobile_sim import drive_with_arm_extended

    d = design("mobile_manipulator_design_round1")
    d["modules"][0] |= {"track_width_m": 0.12, "wheelbase_m": 0.14, "chassis_size_m": [0.16, 0.08, 0.06]}
    d["modules"][1]["mount_xy_m"] = [0.06, 0.0]
    r = drive_with_arm_extended(spec_of(d), req_of("mobile_manipulator"), CATALOG).result
    assert not r.passed and r.cause == "unstable"
    assert "max tilt" in r.message and "wider track" in r.message


def test_sim_tests_for_selects_by_robot_type():
    from taskbot.repair import sim_tests_for
    from taskbot.sim_tests import SIM_TESTS

    mobile = [t.__name__ for t in sim_tests_for(spec_of(design("mobile_design_round2")), req_of("mobile"))]
    assert mobile == ["drive_straight", "turn_in_place", "incline"]
    mm = [t.__name__ for t in sim_tests_for(spec_of(design("mobile_manipulator_design_round2")),
                                            req_of("mobile_manipulator"))]
    assert mm[:4] == ["drive_straight", "turn_in_place", "incline", "drive_with_arm_extended"]
    assert mm[4:] == [t.__name__ for t in SIM_TESTS]  # arm tests, base held fixed


# --------------------------------------------------------------------------- fake backend, pipeline


@pytest.mark.parametrize("prompt, kind", [
    ("a robot that carries a 2 kg box across my office", "mobile"),
    ("a robot that picks toys off the floor and drops them in a basket", "mobile_manipulator"),
    ("Water the three potted plants on my windowsill", "arm"),
    ("a drone that flies my mail to the neighbour", "out_of_scope"),
    ("a robot that swims laps in my pool", "out_of_scope"),
])
def test_fake_scenarios(prompt, kind):
    assert scenario(prompt) == kind


@pytest.mark.parametrize("prompt, first_failures", [
    ("a robot that carries a 2 kg box across my office", {"wheel_torque", "top_speed"}),
    ("a robot that picks toys off the floor and drops them in a basket", {"tip_over"}),
])
def test_fake_mobile_runs_fail_round_one_and_pass_round_two(tmp_path, prompt, first_failures):
    from taskbot.pipeline import run_pipeline

    events = []
    catalog = supported_motors(CATALOG)
    result = run_pipeline(fake_client(catalog), prompt, catalog, tmp_path, record=False,
                          on_event=events.append)
    rounds = result.outcome.rounds
    assert result.passed and len(rounds) == 2
    assert {r.check for r in rounds[0].failures} == first_failures
    assert any(e.startswith("  FAIL tip_over (cause: unstable)") for e in events) == ("tip_over" in first_failures)
    report = (result.out_dir / "report.md").read_text()
    assert "battery_mass_kg not given" in report
    if "toys" in prompt:
        assert "run with the wheeled base fixed in place" in report


def test_out_of_scope_stops_without_a_design(tmp_path):
    from taskbot.cli import main

    code = main(["design", "a drone that flies my mail to the neighbour", "--backend", "fake",
                 "--out", str(tmp_path)])
    assert code == 3
    (out,) = tmp_path.iterdir()
    assert sorted(p.name for p in out.iterdir()) == ["report.md", "requirements.json", "run.json"]
    report = (out / "report.md").read_text()
    assert "OUT OF SCOPE: no design was made" in report and "fly" in report


# --------------------------------------------------------------------------- GLB and app


def test_export_glb_writes_a_viewable_model(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from taskbot.export3d import export_glb

    path = export_glb(spec_of(design("mobile_manipulator_design_round2")), tmp_path / "robot.glb")
    assert path.read_bytes()[:4] == b"glTF"
    scene = trimesh.load(path)
    lo, hi = scene.bounds
    assert hi[1] - lo[1] > 0.3  # y is up in glTF: the arm stands above the chassis


def test_app_design_runs_with_the_fake_backend(tmp_path, monkeypatch):
    pytest.importorskip("trimesh")
    from taskbot import app

    monkeypatch.setattr(app, "OUT_ROOT", tmp_path)
    yields = list(app.design("a robot that carries a 2 kg box across my office", "fake", False))
    log, summary, glb, gifs, requirements, controller, files = yields[-1]
    assert len(yields) > 3  # the log streamed
    assert "Round 1" in log and "FAIL wheel_torque (cause: speed_limited)" in log
    assert summary.startswith("### PASSED")
    assert glb.endswith("robot.glb") and gifs == []
    assert "Requirements (mobile)" in requirements and "battery_mass_kg not given" in requirements
    assert "pololu_37d_50" in controller
    assert {p.rsplit("/", 1)[-1] for p in files[:3]} == {"spec.json", "robot.urdf", "report.md"}
    assert files[3].endswith(".zip")


def test_app_out_of_scope_shows_why(tmp_path, monkeypatch):
    from taskbot import app

    monkeypatch.setattr(app, "OUT_ROOT", tmp_path)
    log, summary, glb, *_ = list(app.design("a drone that flies my mail", "fake", False))[-1]
    assert summary.startswith("### OUT OF SCOPE") and glb is None and "Stopped" in log
