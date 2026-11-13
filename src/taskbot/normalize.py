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

ARM_FIELDS = ("reach_m", "workspace", "dof_min")
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


def _workspace(ws, robot_type: str, notes: list[str], shrunk: list[str]) -> dict:
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
    for k in out:
        lo, hi = (0.0, WORKSPACE_LIMIT_M) if k.startswith("z") else (-WORKSPACE_LIMIT_M, WORKSPACE_LIMIT_M)
        if not lo <= out[k] <= hi:
            v = min(max(out[k], lo), hi)
            text = f"workspace {k} {out[k]:g} m is outside {lo:g} to {hi:g} m: clipped to {v:g} m"
            notes.append(text)
            if k != "z_min" or out[k] > hi:  # below the floor is no target; anything else is cut off
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
        d["workspace"] = _workspace(d.get("workspace"), d["robot_type"], notes, shrunk)
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
    return d, notes


# Words in the task that make a slope or an outdoor route plausible; without one, the model's slope
# is an invention (smoke runs 2026-11-07 and -12: "a flat surface, such as a carpeted floor, with
# a maximum incline of 15 degrees" for picking toys) and an indoor robot gets INDOOR_MAX_INCLINE_DEG.
_SLOPE_WORDS = re.compile(r"\b(ramps?|slopes?|sloped|sloping|hills?|hilly|inclines?|inclined|steps?|thresholds?|"
                          r"outdoors?|outside|gardens?|yards?|backyard|driveways?|kerbs?|curbs?|uphill|downhill|"
                          r"lawns?|paths?|pavements?|sidewalks?|stairs?)\b")
INDOOR_MAX_INCLINE_DEG = 5.0  # door thresholds and gentle indoor ramps (design convention, unverified)
# Items dropped into a separate container: the robot doesn't carry them on its body...
_INTO_CONTAINER = re.compile(r"\b(drops?|dropping|puts?|putting|places?|placing|toss(es)?|throws?|sorts?)\b.*?"
                             r"\b(in|into|onto)\s+(an?|the|its|my|their)?\s*(basket|bin|box|jar|bucket|container|"
                             r"hamper|crate|tub)s?\b")
# ...unless the task says it does.
_CARRIES = re.compile(r"\b(carr(y|ies|ying)|deliver\w*|transport\w*|bring\w*|brings|fetch\w*|haul\w*|"
                      r"on (its|the) (back|top|body|base))\b")


def _uninvent(d: dict, task: str, notes: list[str]) -> None:
    """Mobile requirements the task's own words don't support: a slope steeper than an indoor
    threshold, and a load on the base when the items go into a separate container."""
    text = (task or "").lower()
    if not text.strip():
        return
    if d["max_incline_deg"] > INDOOR_MAX_INCLINE_DEG and not _SLOPE_WORDS.search(text):
        notes.append(f"max_incline_deg {d['max_incline_deg']:g}° reduced to {INDOOR_MAX_INCLINE_DEG:g}°: the task "
                     "names no ramp, slope, step, threshold or outdoor area, so the robot meets only indoor "
                     f"thresholds (indoor convention {INDOOR_MAX_INCLINE_DEG:g}°)")
        d["max_incline_deg"] = INDOOR_MAX_INCLINE_DEG
    if (d.get("base_payload_kg") or 0) > 0 and _INTO_CONTAINER.search(text) and not _CARRIES.search(text):
        notes.append(f"base_payload_kg {d['base_payload_kg']:g} kg set to 0: the task drops the items into a "
                     "separate container and says nothing about carrying anything on the robot")
        d["base_payload_kg"], d["base_payload_bare_kg"] = 0.0, None


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
