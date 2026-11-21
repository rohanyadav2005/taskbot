"""Task 12: what the task's own words say about the scene (taskbot.scene) and the normaliser rules
built on it, from the first real benchmark (qwen2.5:7b-instruct, 2026-11-20)."""

import pytest

from taskbot import scene
from taskbot.normalize import normalize_requirements
from taskbot.requirements import Requirements

ARM = {"task_summary": "t", "robot_type": "arm", "gripper_needed": True, "gripper_kind": "parallel", "dof_min": 3,
       "position_tolerance_m": 0.005, "tolerance_reason": "r",
       "workspace": {"x_min": 0.15, "x_max": 0.35, "y_min": -0.15, "y_max": 0.15, "z_min": 0.0, "z_max": 0.2}}
MM = ARM | {"robot_type": "mobile_manipulator", "terrain": "flat", "speed_m_s": 0.3, "max_incline_deg": 5,
            "base_payload_bare_kg": 0.0,
            "workspace": {"x_min": 0.2, "x_max": 0.45, "y_min": -0.2, "y_max": 0.2, "z_min": 0.0, "z_max": 0.3}}


def norm(data, task):
    d, notes = normalize_requirements(data, task)
    Requirements(**d)  # still valid
    return d, notes


@pytest.mark.parametrize("task, words, kg", [
    ("Move the pieces on a chess board on my table", "pieces on a chess board", 0.03),
    ("move my chess pieces", "chess pieces", 0.03),
    ("Sort a pile of LEGO bricks on my desk", "lego bricks", 0.003),
    ("hand me my reading glasses", "reading glasses", 0.03),   # the longest match at a place wins
    ("Hold a soldering iron over the board", "soldering iron", 0.05),
    ("Collect tennis balls from the court", "tennis balls", 0.06),
    ("Clear empty cups from the tables", "empty cups", 0.3),
    ("Scoop dry cat food from a bin", "cat food", 0.1),
])
def test_everyday_objects(task, words, kg):
    assert scene.object_in(task)[:2] == (words, kg)


def test_unknown_object_and_stated_mass():
    assert scene.object_in("lift the anvil") is None
    assert scene.states_a_mass("my 1.5 kg pan") and scene.states_a_mass("a 500 g bag")
    assert not scene.states_a_mass("four trays")


def test_payload_with_the_gripper_counted_is_lowered_to_the_object():
    """qwen: 0.5 kg for a chess piece "including the gripper"; TaskBot adds the gripper itself."""
    d, notes = norm(ARM | {"payload_bare_kg": 0.4}, "Move the pieces on a chess board on my table")
    assert d["payload_kg"] == 0.037  # 0.03 kg x 1.25, rounded to 1 g
    assert any(n.startswith("payload_bare_kg 0.4 kg lowered to 0.03 kg: the task names 'pieces on a chess board'")
               for n in notes)


@pytest.mark.parametrize("task, bare", [
    ("Lift my 1.5 kg cast iron pan with its lid of cups", 0.4),  # a stated mass: kept
    ("Clear empty cups from the tables", 0.4),                    # within 2x of 0.3 kg: kept
    ("Lift the anvil", 0.4),                                      # no known object: kept
])
def test_payload_kept(task, bare):
    d, _ = norm(ARM | {"payload_bare_kg": bare}, task)
    assert d["payload_kg"] == pytest.approx(bare * 1.25)


def test_fixed_place_makes_a_mobile_manipulator_an_arm():
    d, notes = norm(MM | {"payload_bare_kg": 0.03}, "Move the pieces on a chess board on my table")
    assert d["robot_type"] == "arm" and d["terrain"] is None
    assert any(n.startswith("robot_type 'mobile_manipulator' read as 'arm': the task works at one fixed place")
               for n in notes)
    d, _ = norm(MM | {"payload_bare_kg": 0.06}, "Clear the cups on the table and bring them to the kitchen")
    assert d["robot_type"] == "mobile_manipulator"  # it travels


def test_terrain_comes_from_the_places_the_task_names():
    """Benchmark: an office was "outdoor" (so 0.08 m wheels and no turning in place), a paved garden
    path "flat". Indoor places are flat (carpet only when the task says carpet); any outdoor place
    makes it outdoor; no place word leaves the model's terrain."""
    mobile = {"task_summary": "t", "robot_type": "mobile", "terrain": "outdoor", "speed_m_s": 0.5,
              "max_incline_deg": 15, "base_payload_bare_kg": 2, "position_tolerance_m": 0.05, "tolerance_reason": "r"}
    d, notes = norm(mobile, "a robot that carries a 2 kg box across my office")
    assert (d["terrain"], d["max_incline_deg"]) == ("flat", 5.0)
    assert "terrain 'outdoor' read as 'flat': the task names an indoor place and nowhere outside" in notes
    d, _ = norm(mobile, "carry a 2 kg box across the carpeted living room")
    assert d["terrain"] == "carpet"
    for task in ("Carry 10 kg of potting soil along the paved path from my garage to the vegetable beds",
                 "Collect tennis balls from the court", "carry a 2 kg bag of leaves across the lawn"):
        d, notes = norm(mobile | {"terrain": "flat"}, task)
        assert d["terrain"] == "outdoor" and d["max_incline_deg"] == 8.0, task
        assert any(n.startswith("terrain 'flat' read as 'outdoor': the task drives outdoors") for n in notes)
    d, _ = norm(mobile, "carry a 2 kg box")
    assert d["terrain"] == "outdoor"  # no place named: the model's terrain stands


def test_robot_type_from_the_tasks_words():
    from taskbot.scene import robot_type_from_words

    for task, want in [
        ("Stir the soup in the pot on my stove", "arm"),
        ("Pick screws out of a parts tray next to my workbench", "arm"),
        ("Hold my phone and slowly pan it across my desk", "arm"),          # across a desk is no travel
        ("Deliver letters between desks on our floor", "mobile"),
        ("Drive around the house at night carrying a camera", "mobile"),
        ("Pick up toys from the floor and drop them in the box", "mobile_manipulator"),
        ("Fetch my keys from the hall table", "mobile_manipulator"),         # fetch travels, even to a table
        ("make my mornings easier", None),                                  # no clear words: the LLM decides
        ("help me reach a high shelf", None),
    ]:
        assert robot_type_from_words(task)[0] == want, task


def test_a_task_that_handles_objects_is_never_a_plain_mobile_robot():
    """mm_remote_fetch: qwen chose "mobile" (no arm) to fetch the TV remote, and it PASSED."""
    mobile = {"task_summary": "t", "robot_type": "mobile", "terrain": "flat", "speed_m_s": 0.2,
              "max_incline_deg": 5, "base_payload_bare_kg": 0.4, "gripper_needed": False,
              "position_tolerance_m": 0.05, "tolerance_reason": "r"}
    d, notes = norm(mobile, "Fetch the TV remote from wherever it was left on the coffee table and bring it to me")
    assert d["robot_type"] == "mobile_manipulator" and d["gripper_needed"] and d["gripper_kind"] == "parallel"
    assert (d["workspace"]["z_min"], d["workspace"]["z_max"]) == (0.0, 0.5)  # coffee table ~0.45 m + 0.05
    assert any(n.startswith("robot_type 'mobile' read as 'mobile_manipulator'") for n in notes)
    # Ambiguous words, but it picks things up: an arm is added all the same.
    d, _ = norm(mobile, "pick up anything I point at")
    assert d["robot_type"] == "mobile_manipulator"


@pytest.mark.parametrize("task, base", [
    ("Carry a full 5 kg laundry basket from my bedroom to the washing machine", 6.25),  # a stated mass it carries
    ("a robot that picks toys off the floor and drops them in a basket", 0.0),         # dropped into a basket
    ("a robot that picks toys off the floor and carries them in its own basket", 2.5),  # the model's 2 kg stands
    ("Drive up and down the aisles at night carrying a camera", 0.625),                 # one camera, not 5 kg
    ("Deliver letters and small parcels between desks", 2.5),                           # two kinds: model's stands
])
def test_base_load_from_the_tasks_words(task, base):
    mobile = {"task_summary": "t", "robot_type": "mobile", "terrain": "flat", "speed_m_s": 0.3,
              "max_incline_deg": 5, "base_payload_bare_kg": 2, "position_tolerance_m": 0.05, "tolerance_reason": "r"}
    if "laundry" in task:
        mobile = mobile | {"base_payload_bare_kg": None}  # qwen left it out; code's default was 0
    d, _ = normalize_requirements(mobile, task)
    assert d["base_payload_kg"] == base


def test_payload_from_the_object_the_model_names():
    d, notes = norm(ARM | {"payload_bare_kg": 0.4, "held_object": "a chess piece", "held_count": 1},
                    "help me play board games")
    assert d["payload_kg"] == 0.037
    assert any("the gripper holds 1 x 'a chess piece'" in n for n in notes)
    d, _ = norm(ARM | {"payload_bare_kg": 0.1, "held_object": "eggs", "held_count": 2}, "move eggs to the tray")
    assert d["payload_bare_kg"] == 0.12
    d, _ = norm(ARM | {"payload_bare_kg": 1.5, "held_object": "the pan"}, "lift my 1.5 kg pan")
    assert d["payload_bare_kg"] == 1.5  # the task states the mass
    d, notes = norm(ARM | {"payload_bare_kg": 0.2, "held_object": "a widget"}, "sort widgets")
    assert d["payload_bare_kg"] == 0.2  # not in the table: the model's number stands


def test_object_table_is_broad():
    assert len(scene.OBJECT_KG) >= 60


def test_no_budget_unless_the_task_states_one():
    d, notes = norm(ARM | {"payload_bare_kg": 0.003, "max_budget_usd": 1000}, "Sort LEGO bricks on my desk")
    assert d["max_budget_usd"] is None and "max_budget_usd 1000 removed: the task states no budget or price" in notes
    d, _ = norm(ARM | {"payload_bare_kg": 0.003, "max_budget_usd": 150}, "Sort LEGO bricks on my desk for under $150")
    assert d["max_budget_usd"] == 150


def test_heights_from_the_floor_are_measured_from_the_counter():
    """qwen: z 0.6-1.2 m for a bin on the counter ("a standard countertop height of 0.8 m")."""
    ws = {"x_min": -0.15, "x_max": 0.15, "y_min": -0.15, "y_max": 0.15, "z_min": 0.6, "z_max": 1.2}
    d, notes = norm(ARM | {"payload_bare_kg": 0.1, "workspace": ws},
                    "Scoop dry cat food from a bin on the counter into my cat's bowl twice a day")
    assert (d["workspace"]["z_min"], d["workspace"]["z_max"]) == (0.0, 0.6)
    assert any(n.startswith("workspace z 0.6 to 1.2 m read as measured from the floor") for n in notes)
    assert d["workspace"]["x_min"] > 0  # then turned and trimmed off the base, as before
    # Low targets on a desk, and an arm with no surface named, are left alone.
    for task, z in (("Sort LEGO bricks on my desk", 0.3), ("hold a lamp near the ceiling", 0.6)):
        d, _ = norm(ARM | {"payload_bare_kg": 0.003, "workspace": dict(ARM["workspace"], z_min=z, z_max=z + 0.2)}, task)
        assert d["workspace"]["z_min"] == z


def test_the_bottom_shelf_is_low():
    ws = dict(MM["workspace"], z_min=0.2, z_max=1.2)
    task = "Restock packets of crisps on the bottom shelf of a corner shop from a box on the floor"
    d, notes = norm(MM | {"payload_bare_kg": 0.05, "workspace": ws}, task)
    assert (d["workspace"]["z_min"], d["workspace"]["z_max"]) == (0.2, 0.4)
    assert any(n.startswith("workspace z_max 1.2 m lowered to 0.4 m: the task names the bottom shelf") for n in notes)
    d, _ = norm(MM | {"payload_bare_kg": 0.05, "workspace": ws}, "move crisps from the bottom shelf to the top shelf")
    assert d["workspace"]["z_max"] == 1.2


def test_a_mobile_manipulators_far_targets_are_not_cut_short():
    """mm_tennis_balls: a +-5 m court was marked CUT SHORT; the base drives there."""
    ws = {"x_min": -5, "x_max": 5, "y_min": -5, "y_max": 5, "z_min": 0, "z_max": 0.3}
    d, _ = norm(MM | {"payload_bare_kg": 0.06, "workspace": ws}, "Collect tennis balls from the court")
    assert not any(a.startswith("CUT SHORT") for a in d["assumptions"])
