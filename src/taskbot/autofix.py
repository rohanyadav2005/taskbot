"""Deterministic fixes for failures with a known numeric fix, applied before asking the LLM.

    static_torque / torque_at_speed / a sim test, cause saturated or speed_limited
        -> the cheapest stronger motor on that joint that carries its load, then every joint
           inward re-checked with checks.holding_loads and upgraded where it falls short
    reach / workspace_corners (the chain bends, it is just too short)
        -> lengthen the links after the first pitch joint by the shortfall, split in
           proportion to their lengths, then re-check the motors
    self_collision -> the verified fix that changes the design least (checks.verified_collision_fixes)
    joint_limits   -> widen each short range to the required one, inside ±pi
    structural     -> swapped joint limits, duplicate part names (on raw spec dicts)

Each change is logged as text. autofix_loop applies one round of fixes per iteration, up to
MAX_ITERATIONS, and stops when everything passes, when nothing it knows applies, or when a
fix would produce a design it has already tried. What's left needs a different intent.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field

from taskbot import checks
from taskbot.build_urdf import build_model
from taskbot.checks import CheckResult, holding_loads, usable_torque
from taskbot.repair import same_design
from taskbot.requirements import Requirements
from taskbot.schema import ArmChain, Motor, RobotSpec, supported_motors

MAX_ITERATIONS = 5
TORQUE_CHECKS = {"static_torque", "torque_at_speed"}
REACH_CHECKS = {"reach", "workspace_corners"}
_SIM_SATURATED = re.compile(r"(\w+) (?:saturated|speed-limited) for [\d.]+ s")
DEGENERATE = ("can only spin in place", "None of the", "Only one joint bends the arm")


@dataclass
class AutofixResult:
    spec: RobotSpec
    results: list[CheckResult]
    steps: list[str] = field(default_factory=list)
    rounds: list[tuple[RobotSpec, list[CheckResult]]] = field(default_factory=list)
    stop: str = ""                      # why it stopped


# --------------------------------------------------------------------------- structural


def structural_fix(d: dict) -> tuple[dict, list[str]]:
    """Fix a raw spec dict before validation: swapped limits, duplicate part names."""
    d = json.loads(json.dumps(d))
    steps, seen = [], set()

    def unique(name: str) -> str:
        new, n = name, 2
        while new in seen:
            new, n = f"{name}_{n}", n + 1
        seen.add(new)
        if new != name:
            steps.append(f"renamed duplicate part {name} -> {new}")
        return new

    for m in d.get("modules", []):
        if "name" in m:
            m["name"] = unique(m["name"])
        for seg in m.get("segments", []):
            j = seg.get("joint", {})
            if j.get("lower") is not None and j.get("upper") is not None and j["lower"] > j["upper"]:
                j["lower"], j["upper"] = j["upper"], j["lower"]
                steps.append(f"{j.get('name')}: swapped lower and upper ({j['lower']:g} to {j['upper']:g} rad)")
            for part in (j, seg.get("link", {})):
                if "name" in part:
                    part["name"] = unique(part["name"])
    return d, steps


# --------------------------------------------------------------------------- helpers


def _segments(d: dict) -> list[dict]:
    return [s for m in d["modules"] if m["kind"] == "arm_chain" for s in m["segments"]]


def _position_motors(catalog) -> list[Motor]:
    return sorted((m for m in supported_motors(catalog).values() if "position" in m.modes),
                  key=lambda m: (m.price_usd, m.rated_torque_nm))


def _loads(d: dict, req: Requirements, catalog) -> dict[str, float]:
    spec = RobotSpec.model_validate(d)
    return {jl.joint.name: jl.need for jl in holding_loads(spec, req, build_model(spec, catalog))}


def _upgrade_inward(d: dict, req: Requirements, catalog, steps: list[str], why: str) -> None:
    """Re-check every arm joint (tip inward) against its holding load; upgrade any that falls
    short to the cheapest motor that carries it."""
    motors = _position_motors(catalog)
    for seg in reversed(_segments(d)):
        j = seg["joint"]
        need = _loads(d, req, catalog)[j["name"]]
        current = catalog[j["motor_id"]]
        if usable_torque(current, j["type"]) >= need:
            continue
        pick = next((m for m in motors if usable_torque(m, j["type"]) >= need), None)
        if pick is not None and pick.id != current.id:
            j["motor_id"] = pick.id
            steps.append(f"{j['name']}: {current.id} → {pick.id} (needed {need:.3g} Nm, usable "
                         f"{usable_torque(pick, j['type']):.3g} Nm; {why})")


def _joint_of(r: CheckResult, names: set[str]) -> str | None:
    first = r.message.split(" ", 1)[0]
    if first in names:
        return first
    m = _SIM_SATURATED.search(r.message)
    return m[1] if m and m[1] in names else None


# --------------------------------------------------------------------------- fixes


def _fix_torque(d, req, catalog, failures, steps) -> bool:
    names = {s["joint"]["name"] for s in _segments(d)}
    joints = {_joint_of(r, names) for r in failures
              if (r.check in TORQUE_CHECKS or r.cause in ("saturated", "speed_limited"))
              and r.cause in ("saturated", "speed_limited")} - {None}
    if not joints:
        return False
    motors = _position_motors(catalog)
    changed = False
    for seg in _segments(d):
        j = seg["joint"]
        if j["name"] not in joints:
            continue
        current = catalog[j["motor_id"]]
        need = max([r.value for r in failures if r.value is not None and _joint_of(r, names) == j["name"]]
                   + [_loads(d, req, catalog)[j["name"]]])
        stronger = [m for m in motors if m.rated_torque_nm > current.rated_torque_nm]
        pick = next((m for m in stronger if usable_torque(m, j["type"]) >= need), None)
        pick = pick or (min(stronger, key=lambda m: m.rated_torque_nm) if stronger else None)
        if pick is None:
            continue
        j["motor_id"] = pick.id
        changed = True
        steps.append(f"{j['name']}: {current.id} → {pick.id} (needed {need:.3g} Nm, usable "
                     f"{usable_torque(pick, j['type']):.3g} Nm)")
    if changed:
        _upgrade_inward(d, req, catalog, steps, "re-checked after the motor change outboard")
    return changed


def _fix_reach(d, spec, req, catalog, failures, steps) -> bool:
    if not any(r.check in REACH_CHECKS for r in failures):
        return False
    if any(any(k in r.message for k in DEGENERATE) for r in failures if r.check in REACH_CHECKS):
        return False  # the layout can't reach at all: needs a different intent
    segs = _segments(d)
    bends = [i for i, s in enumerate(segs) if s["joint"]["type"] != "prismatic" and s["joint"]["axis"] in ("x", "y")]
    if not bends:
        return False
    stretch, needed = checks.stretch_m(spec), checks.needed_reach_m(req)
    short = needed - stretch + checks.REACH_TOLERANCE_M
    if short <= 0:  # long enough at full stretch: use the worst miss
        short = max((r.value for r in failures if r.check in REACH_CHECKS and r.value is not None), default=0.0)
    if short <= 0:
        return False
    scalable = [s["link"] for s in segs[bends[0]:] if _length(s["link"]) > 0.05]
    total = sum(_length(lk) for lk in scalable)
    if not scalable or total <= 0:
        return False
    parts = []
    for lk in scalable:
        add = short * _length(lk) / total
        _set_length(lk, _length(lk) + add)
        parts.append(f"{lk['name']} +{add * 100:.1f} cm")
    steps.append(f"lengthened the arm by {short * 100:.1f} cm ({', '.join(parts)}): it was "
                 f"{'too short' if needed > stretch else 'missing targets'} by that much")
    _upgrade_inward(d, req, catalog, steps, "the longer arm weighs and levers more")
    return True


def _length(link: dict) -> float:
    return link["size_m"][2] if link["shape"] == "box" else link["length_m"]


def _set_length(link: dict, v: float) -> None:
    if link["shape"] == "box":
        link["size_m"] = [link["size_m"][0], link["size_m"][1], round(v, 4)]
    else:
        link["length_m"] = round(v, 4)


def _fix_collision(d, spec, req, catalog, failures, steps) -> dict | None:
    if not any(r.check == "self_collision" for r in failures):
        return None
    fixes = checks.verified_collision_fixes(spec, req, catalog)
    if not fixes:
        return None
    _, text, fixed = fixes[0]
    change = re.sub(r" \(collisions.*", "", text)
    rate = re.search(r"collisions ([\d.]+%)", text)[1]
    steps.append(f"self_collision: {change} (verified: {rate} of poses collide)")
    return fixed


def _fix_limits(d, req, failures, steps) -> bool:
    if not any(r.check == "joint_limits" for r in failures):
        return False
    need = req.min_joint_range_rad + 1e-3
    changed = False
    for seg in _segments(d):
        j = seg["joint"]
        if j["type"] != "revolute":
            continue
        lo, hi = j["lower"], j["upper"]
        if hi - lo < need - checks.JOINT_RANGE_ROUNDING_RAD:
            grow = need - (hi - lo)
            lo, hi = lo - grow / 2, hi + grow / 2
            if lo < -math.pi:
                lo, hi = -math.pi, -math.pi + need
            if hi > math.pi:
                lo, hi = math.pi - need, math.pi
        lo, hi = max(lo, -math.pi), min(hi, math.pi)
        if (lo, hi) != (j["lower"], j["upper"]):
            steps.append(f"{j['name']}: range {j['lower']:.4g} to {j['upper']:.4g} rad -> {lo:.4g} to {hi:.4g} rad "
                         f"(needs at least {math.degrees(req.min_joint_range_rad):.0f}°)")
            j["lower"], j["upper"] = round(lo, 4), round(hi, 4)
            changed = True
    return changed


def propose(spec: RobotSpec, req: Requirements, results: list[CheckResult], catalog: dict[str, Motor],
            steps: list[str]) -> RobotSpec | None:
    """One round of every applicable fix. None if nothing applies."""
    failures = [r for r in results if not r.passed]
    if not failures or any(r.cause == "infeasible" for r in failures):
        return None
    d = spec.model_dump(exclude_none=True)
    changed = False
    collision = _fix_collision(d, spec, req, catalog, failures, steps)
    if collision is not None:
        d, changed = collision, True
    changed |= _fix_limits(d, req, failures, steps)
    changed |= _fix_torque(d, req, catalog, failures, steps)
    changed |= _fix_reach(d, RobotSpec.model_validate(d), req, catalog, failures, steps)
    if not changed:
        return None
    d, more = structural_fix(d)
    steps.extend(more)
    return RobotSpec.model_validate(d)


def autofix_loop(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor],
                 check: Callable[[RobotSpec], list[CheckResult]], results: list[CheckResult] | None = None,
                 max_iterations: int = MAX_ITERATIONS, on_event: Callable[[str], None] | None = None) -> AutofixResult:
    """Fix, re-check, repeat. `check` runs the checks (and sims) on a spec."""
    emit = on_event or (lambda line: None)
    results = results if results is not None else check(spec)
    out = AutofixResult(spec, results, rounds=[(spec, results)])
    for i in range(1, max_iterations + 1):
        if all(r.passed for r in out.results):
            out.stop = "all checks pass"
            break
        steps: list[str] = []
        new = propose(out.spec, req, out.results, catalog, steps)
        if new is None:
            out.stop = "no automatic fix applies to the remaining failures"
            break
        if any(same_design(new, s) for s, _ in out.rounds):
            out.stop = "the fix would repeat a design already tried"
            break
        for s in steps:
            emit(f"  autofix {i}: {s}")
        out.steps += [f"autofix {i}: {s}" for s in steps]
        out.spec, out.results = new, check(new)
        out.rounds.append((new, out.results))
    else:
        out.stop = f"{max_iterations} automatic fixes"
    return out


def failing_part(r: CheckResult, spec: RobotSpec) -> str | None:
    names = {s.joint.name for m in spec.modules if isinstance(m, ArmChain) for s in m.segments}
    return _joint_of(r, names)
