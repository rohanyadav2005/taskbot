"""Normalisers: the LLM supplies facts about the scene and choices; code corrects and derives.

Each takes the parsed JSON dict of one LLM reply and returns (fixed dict, notes). They run
inside LLMClient.complete_json before validation, on every attempt, so the validators (which
are unchanged) only reject what code can't fix. Every note is shown in report.md under
"Corrections made by code".

Requirements (normalize_requirements):
    unknown fields dropped; numbers given as strings ("0.3 m") parsed; min/max swapped back;
    an arm workspace clipped to WORKSPACE_LIMIT_M and moved off the base: forward (same
    size), or up when raising z_min cuts at most RAISE_MAX_FRACTION of its height; a mobile
    manipulator's low targets moved in front of its default chassis; reach_m derived from the workspace (the farthest corner from
    the plausible shoulder x REACH_MARGIN, rounded up to 1 cm); fields that don't apply to the
    robot_type set to null (a mobile manipulator's reach_m is measured horizontally from the
    chassis centre, as the reach check measures it); missing mobile fields filled with MOBILE_DEFAULTS; gripper_needed
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
CHASSIS_HALF_WIDTH_M = 0.12
CHASSIS_TOP_M = 0.12
RAISE_MAX_FRACTION = 0.25     # raise a workspace off the base only if that cuts <= 25% of its height
DEFAULT_WORKSPACE = {"x_min": 0.15, "x_max": 0.35, "y_min": -0.15, "y_max": 0.15, "z_min": 0.0, "z_max": 0.25}

ARM_FIELDS = ("reach_m", "workspace", "dof_min")
MOBILE_FIELDS = ("speed_m_s", "max_incline_deg", "terrain", "base_payload_kg", "base_payload_bare_kg",
                 "travel_distance_m")
NUMBER_FIELDS = ("reach_m", "payload_kg", "payload_bare_kg", "speed_m_s", "max_incline_deg",
                 "base_payload_kg", "base_payload_bare_kg", "travel_distance_m", "max_budget_usd",
                 "position_tolerance_m", "min_joint_range_rad", "dof_min")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:[eE]-?\d+)?")


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


def _workspace(ws, robot_type: str, notes: list[str]) -> dict:
    from taskbot.requirements import BASE_HALF_WIDTH_M, BASE_REGION_TOP_M

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
            notes.append(f"workspace {k} {out[k]:g} m is outside {lo:g} to {hi:g} m: clipped to {v:g} m")
            out[k] = v
    if robot_type == "mobile_manipulator":
        # Targets low down must be in front of the chassis (taskbot.sizing's default chassis), not under it.
        front = CHASSIS_FRONT_M + 0.05
        if out["x_min"] < front and out["z_min"] < CHASSIS_TOP_M and out["y_min"] < CHASSIS_HALF_WIDTH_M \
                and out["y_max"] > -CHASSIS_HALF_WIDTH_M:
            shift = front - out["x_min"]
            out["x_min"], out["x_max"] = round(out["x_min"] + shift, 3), round(out["x_max"] + shift, 3)
            notes.append(f"the workspace reached under the chassis: moved it {shift:.3g} m forward so it starts "
                         f"{front:g} m ahead of the chassis centre (x now {out['x_min']:g} to {out['x_max']:g} m)")
        return out
    b, top = BASE_HALF_WIDTH_M, BASE_REGION_TOP_M
    if out["x_min"] < b and out["x_max"] > -b and out["y_min"] < b and out["y_max"] > -b and out["z_min"] < top:
        clearance = b + 0.02
        shift = clearance - out["x_min"]
        raise_by = top - out["z_min"]
        height = out["z_max"] - out["z_min"]
        # Moving it forward keeps its size; raising z_min cuts targets off, so it is the smaller
        # change only when it removes at most RAISE_MAX_FRACTION of the height.
        if height <= 0 or raise_by > RAISE_MAX_FRACTION * height:
            out["x_min"], out["x_max"] = round(out["x_min"] + shift, 3), round(out["x_max"] + shift, 3)
            notes.append(f"the workspace covered the robot's own base: moved it {shift:.3g} m forward "
                         f"(x now {out['x_min']:g} to {out['x_max']:g} m, same size)")
        else:
            notes.append(f"the workspace covered the robot's own base: raised z_min from {out['z_min']:g} to "
                         f"{top:g} m (cuts {raise_by / height:.0%} of its height; moving it forward would take "
                         f"{shift:.3g} m)")
            out["z_min"] = top
            if out["z_max"] < out["z_min"]:
                out["z_max"] = out["z_min"]
    return out


def normalize_requirements(data: dict, task: str = "") -> tuple[dict, list[str]]:
    from taskbot.requirements import Requirements, Workspace, farthest_corner

    d, notes = copy.deepcopy(data), []
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
    if has_arm:
        d["workspace"] = _workspace(d.get("workspace"), d["robot_type"], notes)
        dof = d.get("dof_min")
        if dof is None or not 1 <= dof <= 7:
            notes.append(f"dof_min {dof!r}: used {DEFAULT_DOF}" if dof is not None else f"dof_min missing: used {DEFAULT_DOF}")
            dof = DEFAULT_DOF
        d["dof_min"] = int(dof)
        ws = Workspace(**d["workspace"])
        if d["robot_type"] == "arm":  # the shoulder is on the base axis: the validator's own rule
            corner, dist, where = farthest_corner("arm", ws)
        else:  # the reach check measures reach_m from the robot's axis (chassis centre), horizontally
            corner = max(((x, y, ws.z_max) for x in (ws.x_min, ws.x_max) for y in (ws.y_min, ws.y_max)),
                         key=lambda c: math.hypot(c[0], c[1]))
            dist, where = math.hypot(corner[0], corner[1]), "horizontally from the robot's axis, where the reach check measures"
        reach = math.ceil(dist * REACH_MARGIN * 100 - 1e-9) / 100
        said = d.get("reach_m")
        if said is None or abs(said - reach) > 0.005:
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
    if a is None:
        d["assumptions"] = []
    elif not isinstance(a, list):
        d["assumptions"] = [str(a)]
    else:
        d["assumptions"] = [str(x) for x in a]
    if d.get("max_budget_usd") is not None and d["max_budget_usd"] <= 0:
        d["max_budget_usd"] = None
    return d, notes


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


def default_intent(req):
    """The intent code uses when the LLM gives none it can use: no LLM needed."""
    from taskbot.intent import DesignIntent

    d, _ = normalize_intent({"rationale": "default intent chosen by code (no usable LLM intent)"}, req)
    return DesignIntent.model_validate(d, context={"requirements": req})
