"""Normalisers: the LLM supplies facts about the scene and choices; code corrects and derives.

Each takes the parsed JSON dict of one LLM reply and returns (fixed dict, notes). They run
inside LLMClient.complete_json before validation, on every attempt, so the validators (which
are unchanged) only reject what code can't fix. Every note is shown in report.md under
"Corrections made by code".

Requirements (normalize_requirements):
    unknown fields dropped; numbers given as strings ("0.3 m") parsed; min/max swapped back;
    an out-of-scope verdict in robot_type or an "OUT_OF_SCOPE:"-like assumption kept as the
    "OUT OF SCOPE:" assumption; a workspace clipped to WORKSPACE_LIMIT_M (a "CUT SHORT:"
    assumption when that cuts targets off); an arm workspace around the base or turret turned and
    trimmed to the same targets in front (_turn_and_trim); a mobile manipulator's workspace cut to
    what its arm covers once the base stops (MM_ARM_DEPTH_M, MM_ARM_HALF_WIDTH_M); reach_m derived
    from the workspace (the farthest corner from TaskBot's own shoulder, requirements.reach_corner,
    x REACH_MARGIN, rounded up to 1 mm); fields that don't apply to the robot_type set to null;
    missing mobile fields filled with MOBILE_DEFAULTS; gripper_needed
    and gripper_kind made consistent; payload_kg = payload_bare_kg x PAYLOAD_MARGIN (and the
    same for the base payload) when the bare value is given; a missing or out-of-range
    position_tolerance_m set to DEFAULT_TOLERANCE_M.
Intent (normalize_intent): robot_type and gripper copied from the requirements; fields that
    don't apply set to null; missing choices given defaults (default_intent needs no LLM).
"""

from __future__ import annotations

import copy
import math
import re

from taskbot import scene
from taskbot.requirements import CUT_SHORT_PREFIX, OUT_OF_SCOPE_PREFIX

PAYLOAD_MARGIN = 1.25          # payload_kg = bare mass x 1.25 (applied here, once)
REACH_MARGIN = 1.10            # reach_m = farthest workspace corner from the shoulder x 1.10
WORKSPACE_LIMIT_M = 1.5        # an arm workspace lies within 1.5 m of the base (|x|, |y|, z)
DEFAULT_TOLERANCE_M = 0.005
DEFAULT_ARM_PAYLOAD_KG = 0.2
DEFAULT_DOF = 3
# Documented indoor defaults for a mobile robot whose requirements leave them out.
MOBILE_DEFAULTS = {"speed_m_s": 0.3, "max_incline_deg": 5.0, "terrain": "flat", "base_payload_kg": 0.0}
# A workspace in front of the base, for an arm whose requirements give none.
# A mobile manipulator's default chassis (taskbot.sizing.CHASSIS_M: 0.30 x 0.24 m, top about 0.1 m up).
CHASSIS_FRONT_M = 0.15
# A mobile manipulator drives to its targets: its arm covers at most this region in front of the
# chassis (design convention, unverified: the base stops within a few cm of where it should).
MM_ARM_DEPTH_M = 0.25
MM_ARM_HALF_WIDTH_M = 0.2
DEFAULT_WORKSPACE = {"x_min": 0.15, "x_max": 0.35, "y_min": -0.15, "y_max": 0.15, "z_min": 0.0, "z_max": 0.25}

ARM_FIELDS = ("reach_m", "workspace", "dof_min", "held_object", "held_count")
MOBILE_FIELDS = ("speed_m_s", "max_incline_deg", "terrain", "base_payload_kg", "base_payload_bare_kg",
                 "travel_distance_m")
NUMBER_FIELDS = ("reach_m", "payload_kg", "payload_bare_kg", "speed_m_s", "max_incline_deg",
                 "base_payload_kg", "base_payload_bare_kg", "travel_distance_m", "max_budget_usd",
                 "position_tolerance_m", "min_joint_range_rad", "dof_min")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")
# "OUT OF SCOPE:", "OUT_OF_SCOPE:", "Out-of-scope -", ... at the start of a string (an assumption
# needs the colon or dash; a robot_type that starts with the words is a verdict either way).
_OUT_OF_SCOPE = re.compile(r"out[\s_-]*of[\s_-]*scope\b\s*[:\-–—]?\s*", re.IGNORECASE)
_OUT_OF_SCOPE_LEAD = re.compile(r"out[\s_-]*of[\s_-]*scope\s*[:\-–—]\s*", re.IGNORECASE)


def _number(v):
    """A number from a number, or the first number in a string ("0.3 m" -> 0.3); else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = _NUMBER.search(v.replace(",", "."))
        return float(m[0]) if m else None
    return None


def _bool(v):
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip().lower() in ("true", "yes", "1"):
        return True
    if isinstance(v, str) and v.strip().lower() in ("false", "no", "0", "none", ""):
        return False
    return None if v is None else bool(v)


def _robot_type(v, d) -> str:
    s = str(v or "").lower()
    if "manip" in s or ("mobile" in s and "arm" in s):
        return "mobile_manipulator"
    if any(w in s for w in ("mobile", "wheel", "drive", "rover", "cart")):
        return "mobile"
    if "arm" in s:
        return "arm"
    has_arm = d.get("workspace") is not None or d.get("reach_m") is not None
    has_mobile = any(d.get(f) is not None for f in ("speed_m_s", "terrain", "base_payload_kg"))
    return "mobile_manipulator" if has_arm and has_mobile else "mobile" if has_mobile else "arm"


def _box(ws: dict) -> str:
    return (f"x {ws['x_min']:g} to {ws['x_max']:g}, y {ws['y_min']:g} to {ws['y_max']:g}, "
            f"z {ws['z_min']:g} to {ws['z_max']:g} m")


def _turn_and_trim(out: dict, clearance: float) -> tuple[dict, str]:
    """A box around the robot, as the same targets in front of it: the base yaw (or the chassis)
    turns the arm to any side, so only distance and height matter. Turn the box a multiple of
    90° so its longest reach from the axis points forward, then move its near edge out to
    `clearance`. The farthest corner keeps its distance from the axis and its height; what is
    cut is inside the robot. A box too small to keep anything is moved forward instead."""
    x0, x1, y0, y1 = out["x_min"], out["x_max"], out["y_min"], out["y_max"]
    turns = {"forward": (x0, x1, y0, y1), "from behind": (-x1, -x0, -y1, -y0),
             "from the left": (y0, y1, -x1, -x0), "from the right": (-y1, -y0, x0, x1)}
    side = max(turns, key=lambda k: turns[k][1] + (1e-9 if k == "forward" else 0.0))
    nx0, nx1, ny0, ny1 = turns[side]
    turned = ("" if side == "forward" else "it was turned 180° to the front, " if side == "from behind"
              else f"it was turned 90° {side} to the front, ")
    if nx1 > clearance + 0.01:
        nx0 = max(nx0, clearance)
        how = (f"{turned}its near edge was moved out to {clearance:g} m from the axis, and its farthest point "
               "keeps its distance and height")
    else:
        nx0, nx1 = clearance, clearance + (nx1 - nx0)
        how = f"{turned}it was moved forward to start {clearance:g} m from the axis (same size)"
    new = {"x_min": round(nx0, 3), "x_max": round(nx1, 3), "y_min": round(ny0, 3), "y_max": round(ny1, 3),
           "z_min": out["z_min"], "z_max": out["z_max"]}
    return new, how


def _workspace(ws, robot_type: str, notes: list[str], shrunk: list[str], task: str = "") -> dict:
    from taskbot.requirements import BASE_HALF_WIDTH_M, Workspace, shoulder_point

    if not isinstance(ws, dict):
        notes.append(f"workspace missing: used the default in front of the base {DEFAULT_WORKSPACE}")
        return dict(DEFAULT_WORKSPACE)
    out = {}
    for k in ("x_min", "x_max", "y_min", "y_max", "z_min", "z_max"):
        v = _number(ws.get(k))
        if v is None:
            v = DEFAULT_WORKSPACE[k]
            notes.append(f"workspace {k} missing: used {v:g} m")
        elif not isinstance(ws.get(k), (int, float)):
            notes.append(f"workspace {k} {ws.get(k)!r} read as {v:g} m")
        out[k] = v
    for a in "xyz":
        if out[f"{a}_min"] > out[f"{a}_max"]:
            out[f"{a}_min"], out[f"{a}_max"] = out[f"{a}_max"], out[f"{a}_min"]
            notes.append(f"workspace {a}_min and {a}_max were reversed: swapped")
    if task:  # heights the task's words decide, before anything is cut short
        out = _scene_heights(out, robot_type, task, notes)
    for k in out:
        lo, hi = (0.0, WORKSPACE_LIMIT_M) if k.startswith("z") else (-WORKSPACE_LIMIT_M, WORKSPACE_LIMIT_M)
        if not lo <= out[k] <= hi:
            v = min(max(out[k], lo), hi)
            text = f"workspace {k} {out[k]:g} m is outside {lo:g} to {hi:g} m: clipped to {v:g} m"
            notes.append(text)
            # Below the floor is no target, and a mobile manipulator drives to far x and y (mm_tennis_balls:
            # a +-5 m court); anything else is cut off.
            if k.startswith("z") and (k != "z_min" or out[k] > hi) or (not k.startswith("z") and robot_type != "mobile_manipulator"):
                shrunk.append(f"targets up to {k} = {out[k]:g} m were cut to {v:g} m (TaskBot designs for "
                              f"targets within {WORKSPACE_LIMIT_M:g} m of the base)")
            out[k] = v
    if robot_type == "mobile_manipulator":
        # The base drives to each target, so the arm only covers MM_ARM_DEPTH_M by 2 x MM_ARM_HALF_WIDTH_M
        # in front of the chassis, at the heights the task needs.
        front, hw = CHASSIS_FRONT_M + 0.05, MM_ARM_HALF_WIDTH_M
        new = dict(out)
        if not (out["x_min"] >= front and out["x_max"] <= front + MM_ARM_DEPTH_M + 1e-9):
            depth = min(out["x_max"] - out["x_min"], MM_ARM_DEPTH_M)
            new["x_min"] = round(min(max(out["x_min"], front), front + MM_ARM_DEPTH_M - depth), 3)
            new["x_max"] = round(new["x_min"] + depth, 3)
        if not (out["y_min"] >= -hw - 1e-9 and out["y_max"] <= hw + 1e-9):
            half = min((out["y_max"] - out["y_min"]) / 2, hw)
            new["y_min"], new["y_max"] = round(-half, 3), round(half, 3)
        if new != out:
            notes.append(f"a mobile manipulator drives to each target, so its arm only covers the region in front "
                         f"of the chassis where it works once the base stops: the workspace ({_box(out)}) became "
                         f"{_box(new)} (at most {MM_ARM_DEPTH_M:g} m deep from {front:g} m ahead of the chassis "
                         f"centre and {hw:g} m to each side, at the same heights; the base drives and turns for "
                         "the rest)")
        return new
    b = BASE_HALF_WIDTH_M
    (_, _, shoulder_z), _ = shoulder_point("arm", Workspace(**out))
    if out["x_min"] < b and out["x_max"] > -b and out["y_min"] < b and out["y_max"] > -b and out["z_min"] < shoulder_z:
        new, how = _turn_and_trim(out, b + 0.02)
        notes.append(f"the workspace ({_box(out)}) surrounds the robot's own base and turret, so part of it was "
                     f"inside the robot. The base yaw turns the arm to any side, so {how}: {_box(new)}")
        out = new
    return out


def normalize_requirements(data: dict, task: str = "") -> tuple[dict, list[str]]:
    from taskbot.requirements import Requirements, Workspace, reach_corner

    d, notes, shrunk = copy.deepcopy(data), [], []
    known = set(Requirements.model_fields)
    for k in [k for k in d if k not in known]:
        notes.append(f"ignored unknown field {k!r}")
        del d[k]
    for k in NUMBER_FIELDS:
        if k in d and d[k] is not None and not isinstance(d[k], (int, float)) or isinstance(d.get(k), bool):
            v = _number(d[k])
            notes.append(f"{k} {d[k]!r} read as {v:g}" if v is not None else f"{k} {d[k]!r} isn't a number: ignored")
            d[k] = v
    if not str(d.get("task_summary") or "").strip():
        d["task_summary"] = task or "unnamed task"
        notes.append("task_summary missing: used the task as given")
    rt = d.get("robot_type")
    scope_notes = []
    if isinstance(rt, str) and rt not in ("arm", "mobile", "mobile_manipulator") and _OUT_OF_SCOPE.match(rt.strip()):
        # The model put its out-of-scope verdict in robot_type: keep it where TaskBot looks for it.
        reason = _OUT_OF_SCOPE.sub("", rt.strip(), count=1).strip() or "the model's robot_type said so"
        scope_notes.append(f"{OUT_OF_SCOPE_PREFIX} {reason}")
        notes.append(f"robot_type {rt!r} is an out-of-scope verdict, not a robot type: kept it as an "
                     f"'{OUT_OF_SCOPE_PREFIX}' assumption")
        rt = d["robot_type"] = None
    if rt not in ("arm", "mobile", "mobile_manipulator"):
        d["robot_type"] = _robot_type(rt, d)
        notes.append(f"robot_type {rt!r} read as {d['robot_type']!r}" if rt else
                     f"robot_type missing: {d['robot_type']!r} from the fields given")
    if task and not scope_notes:
        _type_from_words(d, task, notes)
    has_arm = d["robot_type"] in ("arm", "mobile_manipulator")
    mobile = d["robot_type"] in ("mobile", "mobile_manipulator")
    for k in ARM_FIELDS if not has_arm else ():
        if d.get(k) is not None:
            notes.append(f"{k} doesn't apply to a {d['robot_type']} robot: set to null")
        d[k] = None
    for k in MOBILE_FIELDS if not mobile else ():
        if d.get(k) is not None:
            notes.append(f"{k} doesn't apply to an arm: set to null")
        d[k] = None
    # Gripper.
    needed, kind = _bool(d.get("gripper_needed")), d.get("gripper_kind")
    if kind not in ("parallel", "suction"):
        if kind:
            notes.append(f"gripper_kind {kind!r} isn't parallel or suction: used parallel")
            kind = "parallel"
        else:
            kind = None
    if not has_arm:
        if needed or kind:
            notes.append("a mobile robot has no arm: no gripper")
        needed, kind = False, None
    elif needed and kind is None:
        kind = "parallel"
        notes.append("gripper_needed with no gripper_kind: used parallel")
    elif kind and not needed:
        needed = True
        notes.append(f"gripper_kind {kind} given with gripper_needed false: set gripper_needed true")
    elif needed is None:
        needed = False
    d["gripper_needed"], d["gripper_kind"] = bool(needed), kind
    if has_arm and task:
        _typical_payload(d, task, notes)
    # Payloads: the margin is applied here, exactly once, when the bare value is given.
    for bare_key, key, applies in (("payload_bare_kg", "payload_kg", has_arm), ("base_payload_bare_kg", "base_payload_kg", mobile)):
        bare = d.get(bare_key)
        if not applies:
            continue
        if bare is not None and bare >= 0:
            value = round(bare * PAYLOAD_MARGIN, 3)
            if d.get(key) is not None and abs(d[key] - value) > 1e-6:
                notes.append(f"{key} {d[key]:g} kg replaced by {bare:g} kg bare x {PAYLOAD_MARGIN} = {value:g} kg "
                             "(the margin applied once, by code)")
            elif d.get(key) is None:
                notes.append(f"{key} = {bare:g} kg bare x {PAYLOAD_MARGIN} = {value:g} kg")
            d[key] = value
        elif d.get(key) is not None and d[key] < 0:
            notes.append(f"{key} {d[key]:g} kg is negative: used {abs(d[key]):g} kg")
            d[key] = abs(d[key])
    if not has_arm:
        if d.get("payload_kg"):
            notes.append("payload_kg doesn't apply to a mobile robot (it has no gripper): set to 0")
        d["payload_kg"], d["payload_bare_kg"] = 0.0, None
    elif d.get("payload_kg") is None:
        d["payload_kg"] = DEFAULT_ARM_PAYLOAD_KG
        notes.append(f"payload_kg missing: used {DEFAULT_ARM_PAYLOAD_KG} kg")
    if mobile:
        for k, v in MOBILE_DEFAULTS.items():
            if d.get(k) is None:
                d[k] = v
                notes.append(f"{k} missing: used the indoor default {v}")
        if d.get("terrain") not in ("flat", "carpet", "outdoor"):
            notes.append(f"terrain {d['terrain']!r} isn't flat, carpet or outdoor: used flat")
            d["terrain"] = "flat"
        if not 0 <= d["max_incline_deg"] <= 30:
            v = min(max(d["max_incline_deg"], 0.0), 30.0)
            notes.append(f"max_incline_deg {d['max_incline_deg']:g} clipped to {v:g}")
            d["max_incline_deg"] = v
        if d["speed_m_s"] <= 0:
            notes.append(f"speed_m_s {d['speed_m_s']:g} isn't positive: used {MOBILE_DEFAULTS['speed_m_s']}")
            d["speed_m_s"] = MOBILE_DEFAULTS["speed_m_s"]
        if d.get("travel_distance_m") is not None and d["travel_distance_m"] <= 0:
            d["travel_distance_m"] = None
        _uninvent(d, task, notes)
    if has_arm:
        d["workspace"] = _workspace(d.get("workspace"), d["robot_type"], notes, shrunk, task)
        dof = d.get("dof_min")
        if dof is None or not 1 <= dof <= 7:
            notes.append(f"dof_min {dof!r}: used {DEFAULT_DOF}" if dof is not None else f"dof_min missing: used {DEFAULT_DOF}")
            dof = DEFAULT_DOF
        d["dof_min"] = int(dof)
        ws = Workspace(**d["workspace"])
        corner, dist, where = reach_corner(d["robot_type"], ws)  # from the shoulder the sizer uses
        reach = math.ceil(dist * REACH_MARGIN * 1000 - 1e-9) / 1000
        said = d.get("reach_m")
        if said is None or abs(said - reach) > 0.0005:
            text = (f"reach_m {reach:g} m derived from the workspace: its farthest corner "
                    f"({', '.join(f'{v:g}' for v in corner)}) is {dist:.3g} m away ({where}), "
                    f"x {REACH_MARGIN}")
            if said is not None and abs(said - reach) > 0.2 * reach:
                text += f"; the model said {said:g} m"
            notes.append(text)
        d["reach_m"] = reach
    tol = d.get("position_tolerance_m")
    if tol is None or not 0 < tol <= 0.1:
        notes.append(f"position_tolerance_m {tol!r} is missing or outside (0, 0.1] m: used {DEFAULT_TOLERANCE_M} m")
        d["position_tolerance_m"] = DEFAULT_TOLERANCE_M
    if not str(d.get("tolerance_reason") or "").strip():
        d["tolerance_reason"] = "not given; TaskBot used the requirements' tolerance"
        notes.append("tolerance_reason missing")
    mjr = d.get("min_joint_range_rad")
    if mjr is not None and not 0 < mjr <= 2 * math.pi:
        notes.append(f"min_joint_range_rad {mjr:g} is outside (0, 2π]: used the default 90°")
        d.pop("min_joint_range_rad")
    a = d.get("assumptions")
    a = [] if a is None else [str(x) for x in a] if isinstance(a, list) else [str(a)]
    for i, text in enumerate(a):
        m = _OUT_OF_SCOPE_LEAD.match(text.strip())
        if m and not text.strip().startswith(OUT_OF_SCOPE_PREFIX):
            a[i] = f"{OUT_OF_SCOPE_PREFIX} {text.strip()[m.end():].strip()}"
            notes.append(f"assumption starting {text.strip()[:m.end()]!r} read as '{OUT_OF_SCOPE_PREFIX}'")
    a = [x for x in a if not x.strip().startswith(CUT_SHORT_PREFIX)]  # code's own marker: only code adds it
    d["assumptions"] = a + [n for n in scope_notes if n not in a] + [f"{CUT_SHORT_PREFIX} {x}" for x in shrunk]
    if d.get("max_budget_usd") is not None and d["max_budget_usd"] <= 0:
        d["max_budget_usd"] = None
    if d.get("max_budget_usd") is not None and task and not scene.states_a_budget(task):
        notes.append(f"max_budget_usd {d['max_budget_usd']:g} removed: the task states no budget or price")
        d["max_budget_usd"] = None
    return d, notes


# Without a slope word in the task (scene.SLOPE), the model's slope is an invention (smoke runs
# 2026-11-07 and -12: 15° for picking toys) and is capped by the place the task names.
INDOOR_MAX_INCLINE_DEG = 5.0  # door thresholds and gentle indoor ramps (design convention, unverified)
OUTDOOR_MAX_INCLINE_DEG = 8.0  # garden paths and driveways without a named slope (design convention, unverified)
# Items dropped into a separate container (named in the note when the base load is set to 0).
_INTO_CONTAINER = re.compile(r"\b(drops?|dropping|puts?|putting|places?|placing|toss(es)?|throws?|sorts?)\b.*?"
                             r"\b(in|into|onto)\s+(an?|the|its|my|their)?\s*(basket|bin|box|jar|bucket|container|"
                             r"hamper|crate|tub)s?\b")


def _type_from_words(d: dict, task: str, notes: list[str]) -> None:
    """The robot type the task's own words decide (scene.robot_type_from_words), like the scope gate:
    clear words decide and the change is noted; otherwise the model's type stands, except that a task
    that must pick up or hold something is never a 'mobile' robot (it has no arm)."""
    said = d["robot_type"]
    want, why = scene.robot_type_from_words(task)
    if want is None and said == "mobile" and scene.handles_objects(task):
        hq = ", ".join(f'"{w}"' for w in dict.fromkeys(scene.handles_objects(task)))
        want, why = "mobile_manipulator", f"a mobile robot has no arm, and the task handles objects ({hq})"
    if want is None or want == said:
        return
    notes.append(f"robot_type {said!r} read as {want!r}: {why} (TaskBot's robot-type words)")
    d["robot_type"] = want
    if want in ("arm", "mobile_manipulator") and said == "mobile":
        # The reply described a robot without an arm: fill what the arm needs from the task's words.
        if not d.get("gripper_needed"):
            d["gripper_needed"], d["gripper_kind"] = True, d.get("gripper_kind") or "parallel"
            notes.append("gripper_needed set true (parallel): the task handles objects")
        if not isinstance(d.get("workspace"), dict) and want == "mobile_manipulator":
            heights = scene.surface_heights(task)
            top = max([z for _, z in heights], default=0.0)
            d["workspace"] = {"x_min": 0.2, "x_max": 0.45, "y_min": -0.2, "y_max": 0.2,
                              "z_min": 0.0, "z_max": round(max(top + ON_SURFACE_Z_M, 0.25), 3)}
            named = ", ".join(f"{w} ~{z:g} m" for w, z in heights)
            notes.append(f"workspace set to {_box(d['workspace'])}: in front of the chassis, from the floor to "
                         + (f"{ON_SURFACE_Z_M:g} m above the highest surface the task names ({named}; design "
                            "conventions, unverified)" if heights else "0.25 m (no surface named)"))


def _typical_payload(d: dict, task: str, notes: list[str]) -> None:
    """The gripper's payload from TaskBot's table of everyday objects (scene.OBJECT_KG), unless the
    task states a mass itself. The object the model says the gripper holds (held_object x
    held_count) sets it; with no such object, a payload more than MAX_RATIO times the heaviest object
    the task names is lowered to it (the model counted the gripper, several objects or a guess)."""
    bare = d.get("payload_bare_kg")
    if bare is None and d.get("payload_kg") is not None:
        bare = d["payload_kg"] / PAYLOAD_MARGIN
    count = d.get("held_count")
    count = int(count) if isinstance(count, (int, float)) and not isinstance(count, bool) and 1 <= count <= 50 else 1
    if d.get("held_count") is not None and d["held_count"] != count:
        notes.append(f"held_count {d['held_count']!r} isn't 1 to 50: used {count}")
    if d.get("held_count") is not None:
        d["held_count"] = count
    if scene.states_a_mass(task):
        return
    named = scene.object_named(str(d.get("held_object") or ""))
    if named is not None:
        words, kg, what = named
        new = round(kg * count, 4)
        if bare is None or abs(bare - new) > 0.01 * new:
            notes.append(f"payload_bare_kg {'missing' if bare is None else f'{bare:.3g} kg'} set to {new:g} kg: the "
                         f"gripper holds {count} x {d.get('held_object')!r} ({what}, about {kg:g} kg each in "
                         "TaskBot's table of everyday objects, unverified); TaskBot adds the gripper's own mass")
            d["payload_bare_kg"], d["payload_kg"] = new, None
        return
    found = scene.objects_in(task)
    if not found or bare is None:
        return
    words, kg, what = max(found, key=lambda f: f[1])
    if bare <= scene.MAX_RATIO * kg:
        return
    notes.append(f"payload_bare_kg {bare:.3g} kg lowered to {kg:g} kg: the task names {words!r}, and {what} weighs "
                 f"about {kg:g} kg (TaskBot's table of everyday objects, unverified); the gripper holds one at a "
                 "time and TaskBot adds the gripper's own mass itself")
    d["payload_bare_kg"], d["payload_kg"] = kg, None


def _scene_heights(ws: dict, robot_type: str, task: str, notes: list[str]) -> dict:
    """Target heights the task's words contradict: a fixed arm on a desk or counter with every target
    at desk height or above (measured from the floor, not from the surface the arm stands on), and
    targets far above "the bottom shelf"."""
    if robot_type == "arm" and scene.on_a_surface(task) and ws["z_min"] >= scene.SURFACE_FLOOR_Z_M:
        dz = ws["z_min"]
        new = dict(ws, z_min=0.0, z_max=round(ws["z_max"] - dz, 3))
        notes.append(f"workspace z {ws['z_min']:g} to {ws['z_max']:g} m read as measured from the floor: the arm "
                     f"stands on the surface the task names, and TaskBot measures heights from the arm's base, so "
                     f"every target was lowered by {dz:g} m to z {new['z_min']:g} to {new['z_max']:g} m (the lowest "
                     f"at the surface; a target {scene.SURFACE_FLOOR_Z_M:g} m or more above a desk or counter top "
                     "is the top's own height from the floor)")
        ws = new
    ws = _invented_heights(ws, robot_type, task, notes)
    if scene.bottom_shelf_only(task) and ws["z_max"] > scene.BOTTOM_SHELF_Z_M:
        cap = scene.BOTTOM_SHELF_Z_M
        new = dict(ws, z_max=cap, z_min=min(ws["z_min"], round(cap / 2, 3)))
        notes.append(f"workspace z_max {ws['z_max']:g} m lowered to {cap:g} m: the task names the bottom shelf "
                     f"and no higher one, and a bottom shelf is at most about {cap:g} m up (design convention, "
                     "unverified)")
        ws = new
    return ws


# Targets the task gives no height for (design conventions, unverified): a fixed arm works at most
# this high above the surface it stands on; a mobile manipulator up to just above the surfaces the
# task names (things stand at most this high on them), or up to a table or counter top when it names none.
ARM_UNNAMED_Z_MAX_M = 0.5
ON_SURFACE_Z_M = 0.2
MM_UNNAMED_Z_MAX_M = 1.0


def _invented_heights(ws: dict, robot_type: str, task: str, notes: list[str]) -> dict:
    """Heights the task's words don't support: targets far above anything the task names, and, for a
    mobile manipulator working on raised surfaces only, targets far below them."""
    if scene.names_a_height(task):
        return ws
    surfaces = [(w, z) for w, z in scene.surface_heights(task) if z > 0]
    floor = any(z == 0 for _, z in scene.surface_heights(task))
    if robot_type == "mobile_manipulator":
        cap = (max(z for _, z in surfaces) + ON_SURFACE_Z_M) if surfaces else MM_UNNAMED_Z_MAX_M
        named = (", ".join(f"{w} ~{z:g} m" for w, z in surfaces) + f", plus {ON_SURFACE_Z_M:g} m for what stands "
                 "on them") if surfaces else "no surface or height, so up to a table or counter top"
    else:
        cap, named = ARM_UNNAMED_Z_MAX_M, "no height, so a desk-scale reach above the surface the arm stands on"
    new = dict(ws)
    if ws["z_max"] > cap + 1e-9:
        new["z_max"] = round(cap, 3)
        new["z_min"] = min(ws["z_min"], round(cap / 2, 3))
    if robot_type == "mobile_manipulator" and surfaces and not floor:
        low = min(z for _, z in surfaces) - 0.1
        if new["z_min"] < low < new["z_max"]:
            new["z_min"] = round(low, 3)
    if new != ws:
        notes.append(f"workspace z {ws['z_min']:g} to {ws['z_max']:g} m became z {new['z_min']:g} to {new['z_max']:g} m: "
                     f"the task names {named} (design conventions, unverified), and no high shelf, wall or other "
                     "height")
    return new


def _uninvent(d: dict, task: str, notes: list[str]) -> None:
    """Mobile requirements from the task's own words: the terrain of the places it names, a slope no
    steeper than those places have, and the load on the base (_base_load)."""
    text = (task or "").lower()
    if not text.strip():
        return
    terrain, words = scene.terrain_from_words(text)
    if terrain is not None and d.get("terrain") != terrain:
        quoted = ", ".join(f'"{w}"' for w in dict.fromkeys(words))
        why = {"outdoor": f"the task drives outdoors ({quoted})",
               "carpet": f"the task names carpet ({quoted})",
               "flat": "the task names an indoor place and nowhere outside"}[terrain]
        if terrain == "flat" and scene.court_only(words):
            why = (f"the task's place is a sports court ({quoted}), a hard, level surface like a concrete floor "
                   "(grass or clay would be named)")
        notes.append(f"terrain {d.get('terrain')!r} read as {terrain!r}: {why}")
        d["terrain"] = terrain
    cap = OUTDOOR_MAX_INCLINE_DEG if scene.outdoors(text) else INDOOR_MAX_INCLINE_DEG
    if d["max_incline_deg"] > cap and not scene.names_a_slope(text):
        where = ("an outdoor route but no ramp, slope, hill or step, so the robot meets only the gentle grades of "
                 f"paths and driveways (outdoor convention {cap:g}°)" if cap == OUTDOOR_MAX_INCLINE_DEG else
                 "no ramp, slope, step, threshold or outdoor area, so the robot meets only indoor thresholds "
                 f"(indoor convention {cap:g}°)")
        notes.append(f"max_incline_deg {d['max_incline_deg']:g}° reduced to {cap:g}°: the task names {where}")
        d["max_incline_deg"] = cap
    _base_load(d, text, notes)


# A plural object on the base: at most this many of them (design convention, unverified).
MAX_ITEMS_ON_BASE = 10


def _base_load(d: dict, text: str, notes: list[str]) -> None:
    """The load on the base from the task's words: none unless the task says the robot carries
    something (carry, bring, deliver, transport, "a full ..."); a stated mass it carries is the load;
    otherwise the object's typical mass (scene.OBJECT_KG), up to MAX_ITEMS_ON_BASE of a plural."""
    base = d.get("base_payload_kg") or 0.0
    carry = scene.carries(text)
    if not carry:
        if base > 0:
            why = ("drops the items into a separate container and says nothing about carrying anything on the "
                   "robot" if _INTO_CONTAINER.search(text) else "says nothing about carrying anything on the robot")
            notes.append(f"base_payload_kg {base:g} kg set to 0: the task {why}")
            d["base_payload_kg"], d["base_payload_bare_kg"] = 0.0, None
        return
    bare = d.get("base_payload_bare_kg")
    bare = base / PAYLOAD_MARGIN if bare is None else bare
    quoted = ", ".join(f'"{w}"' for w in dict.fromkeys(carry))
    stated = scene.stated_kg(text)
    kinds = scene.objects_in(text)
    found = kinds[0] if len(kinds) == 1 else None  # several kinds (letters and parcels): the model's guess stands
    new, why = None, ""
    if stated is not None:
        if bare < 0.9 * stated:
            new, why = stated, f"the task carries ({quoted}) a stated {stated:g} kg"
    elif found is not None:
        words, kg, what = found
        n = MAX_ITEMS_ON_BASE if words.endswith("s") else 1
        if bare <= 0:
            new, why = kg * n, (f"the task carries ({quoted}) {words!r}: {what} weighs about {kg:g} kg"
                                + (f", up to {n} of them" if n > 1 else "") + " (TaskBot's table, unverified)")
        elif bare > scene.MAX_RATIO * kg * n:
            new, why = kg * n, (f"the task carries {words!r} and states no mass: {what} weighs about {kg:g} kg"
                                + (f", up to {n} of them" if n > 1 else "") + " (TaskBot's table, unverified)")
    if new is None:
        return
    value = round(new * PAYLOAD_MARGIN, 3)
    notes.append(f"base_payload_bare_kg {bare:.3g} kg set to {new:.3g} kg: {why}; base_payload_kg = "
                 f"{new:.3g} x {PAYLOAD_MARGIN} = {value:g} kg")
    d["base_payload_bare_kg"], d["base_payload_kg"] = round(new, 4), value


def normalize_intent(data: dict, req) -> tuple[dict, list[str]]:
    from taskbot.intent import DesignIntent

    d, notes = copy.deepcopy(data) if isinstance(data, dict) else {}, []
    for k in [k for k in d if k not in DesignIntent.model_fields]:
        notes.append(f"intent: ignored unknown field {k!r}")
        del d[k]
    for key, value in (("robot_type", req.robot_type),
                       ("gripper", req.gripper_kind if req.gripper_needed else "none")):
        if d.get(key) is not None and d[key] != value:
            notes.append(f"intent {key} {d[key]!r} replaced by the requirements' {value!r}")
        d[key] = value
    has_arm, mobile = req.has_arm, req.is_mobile
    layouts = ("yaw_pitch_pitch", "yaw_pitch_pitch_pitch", "yaw_pitch_pitch_wristroll")
    if not has_arm:
        d["arm_layout"] = None
    elif d.get("arm_layout") not in layouts:
        if d.get("arm_layout"):
            notes.append(f"intent arm_layout {d['arm_layout']!r} isn't one of {', '.join(layouts)}: used yaw_pitch_pitch_pitch")
        else:
            notes.append("intent arm_layout missing: used yaw_pitch_pitch_pitch")
        d["arm_layout"] = "yaw_pitch_pitch_pitch"
    if d.get("link_material") not in ("auto", "pla", "aluminium"):
        if d.get("link_material"):
            notes.append(f"intent link_material {d['link_material']!r}: used auto")
        d["link_material"] = "auto"
    if not mobile:
        for k in ("drive", "caster", "arm_mount"):
            if d.get(k) is not None:
                notes.append(f"intent {k} doesn't apply to an arm: set to null")
            d[k] = None
    else:
        if d.get("drive") not in ("differential", "four_wheel"):
            drive = "four_wheel" if req.terrain == "outdoor" else "differential"
            notes.append(f"intent drive {d.get('drive')!r}: used {drive} for {req.terrain} terrain"
                         if d.get("drive") else f"intent drive missing: used {drive} for {req.terrain} terrain")
            d["drive"] = drive
        caster = _bool(d.get("caster"))
        if d["drive"] == "differential":
            if caster is not True:
                notes.append("intent: a differential base gets a caster")
            d["caster"] = True
        else:
            d["caster"] = None
        if req.robot_type == "mobile_manipulator":
            if d.get("arm_mount") not in ("front", "centre"):
                notes.append(f"intent arm_mount {d.get('arm_mount')!r}: used front" if d.get("arm_mount")
                             else "intent arm_mount missing: used front")
                d["arm_mount"] = "front"
        else:
            d["arm_mount"] = None
    nm = d.get("not_modelled")
    d["not_modelled"] = [str(x) for x in nm] if isinstance(nm, list) else ([str(nm)] if nm else [])
    r = str(d.get("rationale") or "").strip()
    if not r:
        notes.append("intent rationale missing")
        r = "not given"
    d["rationale"] = r[:600]
    return d, notes


def _intent_key(intent) -> tuple:
    return (intent.arm_layout, intent.drive, intent.arm_mount, intent.gripper)


def alternative_intent(req, tried: list):
    """The next intent code would try when the LLM gives no usable revision: the other drive, then
    the other arm layouts and mounts, in a fixed order, skipping any already tried. None when every
    combination has been tried."""
    from itertools import product

    from taskbot.intent import DesignIntent

    layouts = ("yaw_pitch_pitch_pitch", "yaw_pitch_pitch", "yaw_pitch_pitch_wristroll") if req.has_arm else (None,)
    drives = ("differential", "four_wheel") if req.is_mobile else (None,)
    mounts = ("front", "centre") if req.robot_type == "mobile_manipulator" else (None,)
    seen = {_intent_key(i) for i in tried}
    for mount, layout, drive in product(mounts, layouts, drives):
        d, _ = normalize_intent({"arm_layout": layout, "drive": drive, "arm_mount": mount,
                                 "rationale": "alternative chosen by code (no usable LLM revision)"}, req)
        intent = DesignIntent.model_validate(d, context={"requirements": req})
        if _intent_key(intent) not in seen:
            return intent
    return None


def shift_workspace_forward(req, dx: float):
    """A mobile manipulator's requirements with the arm's targets dx further ahead of the chassis
    (the base stops further back) and reach_m derived again, the same way."""
    from taskbot.requirements import Workspace, reach_corner

    ws = req.workspace
    moved = Workspace(**(ws.model_dump() | {"x_min": round(ws.x_min + dx, 3), "x_max": round(ws.x_max + dx, 3)}))
    _, dist, _ = reach_corner(req.robot_type, moved)
    return req.model_copy(update={"workspace": moved, "reach_m": math.ceil(dist * REACH_MARGIN * 1000 - 1e-9) / 1000})


def default_intent(req):
    """The intent code uses when the LLM gives none it can use: no LLM needed."""
    from taskbot.intent import DesignIntent

    d, _ = normalize_intent({"rationale": "default intent chosen by code (no usable LLM intent)"}, req)
    return DesignIntent.model_validate(d, context={"requirements": req})
