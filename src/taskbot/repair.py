"""design -> build -> check -> simulate -> repair, until everything passes or rounds run out.

Each round runs the static checks first; the (slower) simulation tests run only
once every static check passes.

Repair requests carry, besides the failures: for each check that also failed last
round, whether the last change made it better or worse; a compact history of every
round (what failed, what changed); and, when only reach-type checks fail, a note that
motors can't fix them. They also point out a design identical to an earlier round, a
description claiming changes the spec doesn't have, and reach and torque fixes that pull
against each other.

The loop stops early when a check says no design can pass (cause "infeasible"), or when
the model sends back an earlier design for the second time.

The final design is the best round (fewest failing checks, then the smallest total
normalised shortfall), not necessarily the last. When a round is worse than the best one,
the next repair starts again from the best design.
"""

from __future__ import annotations

from collections.abc import Callable
import re
from dataclasses import dataclass, field
from typing import Any

from taskbot.checks import CheckResult, run_checks
from taskbot.design import design_robot, design_system_prompt
from taskbot.llm import LLMClient, LLMOutputError
from taskbot.prompts import load_prompt
from taskbot.requirements import Requirements
from taskbot.schema import ArmChain, Motor, RobotSpec, load_motor_catalog, supported_motors
from taskbot.sim_tests import SIM_TESTS, SimRun

DEFAULT_MAX_ROUNDS = 5


def _mobile_tests():
    from taskbot.mobile_sim import MOBILE_SIM_TESTS, drive_with_arm_extended

    return MOBILE_SIM_TESTS + [drive_with_arm_extended]


SIM_TEST_NAMES = [t.__name__ for t in SIM_TESTS] + [t.__name__ for t in _mobile_tests()]
Event = Callable[[str], None]


def sim_tests_for(spec: RobotSpec, req: Requirements) -> list:
    """The simulation tests that apply: arm tests (base held fixed for a mobile manipulator)
    and/or driving tests, and the arm-extended driving test for mobile manipulators."""
    from taskbot.mobile_sim import MOBILE_SIM_TESTS, drive_with_arm_extended

    tests = []
    if req.is_mobile and spec.is_mobile:
        tests += MOBILE_SIM_TESTS
        if req.has_arm and spec.has_arm:
            tests.append(drive_with_arm_extended)
    if req.has_arm and spec.has_arm:
        tests += SIM_TESTS
    return tests


def _line(r: CheckResult) -> str:
    status = "PASS" if r.passed else "FAIL"
    cause = f" (cause: {r.cause})" if r.cause else ""
    return f"  {status} {r.check}{cause}: {r.message}"


@dataclass(frozen=True)
class Round:
    number: int
    spec: RobotSpec
    results: list[CheckResult]          # static checks, then sim tests if they ran
    sim_runs: list[SimRun] = field(default_factory=list)

    @property
    def failures(self) -> list[CheckResult]:
        return [r for r in self.results if not r.passed]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class RepairOutcome:
    requirements: Requirements
    max_rounds: int
    rounds: list[Round] = field(default_factory=list)
    stop_reason: str | None = None  # set when the loop ended early
    infeasible: list[str] = field(default_factory=list)  # messages of checks no design can pass

    @property
    def passed(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].passed

    @property
    def final(self) -> Round:
        """The best round: what gets saved and reported as the design."""
        return best_round(self.rounds)

    @property
    def last(self) -> Round:
        return self.rounds[-1]


def failed_checks(rnd: Round) -> int:
    return len({r.check for r in rnd.failures})


def round_rank(rnd: Round) -> tuple[int, float]:
    """(failing checks, total normalised shortfall): lower is better. A failure's shortfall is
    |value - limit| / |limit| (limit floored at 1e-3); failures without numbers count 1."""
    short = 0.0
    for r in rnd.failures:
        if r.value is None or r.limit is None:
            short += 1.0
        else:
            short += abs(r.value - r.limit) / max(abs(r.limit), 1e-3)
    return failed_checks(rnd), short


def best_round(rounds: list[Round]) -> Round:
    """Lowest rank; the later round on a tie."""
    return min(rounds, key=lambda r: (round_rank(r), -r.number))


def format_failures(failures: list[CheckResult]) -> str:
    return "\n".join(f"- [{r.check}]" + (f" (cause: {r.cause})" if r.cause else "") + f" {r.message}"
                     for r in failures)


# --------------------------------------------------------------------------- progress

REACH_CHECKS = {"reach", "workspace_corners"}
MOTOR_NOTE = ("Motor changes won't help this failure. Reach depends only on the joint axes, the link "
              "lengths and the base height.")
# check -> (what its value measures, scale, unit) for progress lines; others use the raw value.
PROGRESS_LABELS = {
    "reach": ("reach miss", 100, "cm"),
    "workspace_corners": ("worst corner miss", 100, "cm"),
    "static_torque": ("torque needed", 1, "Nm"),
    "torque_at_speed": ("gravity torque at speed", 1, "Nm"),
    "wheel_torque": ("wheel torque needed", 1, "Nm"),
    "top_speed": ("top speed", 1, "m/s"),
    "tip_over": ("stability margin", 1000, "mm"),
    "traction": ("traction force needed", 1, "N"),
    "self_collision": ("share of colliding poses", 100, "%"),
    "joint_limits": ("narrowest joint range", 180 / 3.141592653589793, "°"),
}
SAME_WITHIN = 0.02  # a change under 2% of the gap to the limit counts as no change


def _worst(results: list[CheckResult], check: str) -> CheckResult | None:
    """The failing result of `check` furthest from its limit (some checks report one per joint)."""
    bad = [r for r in results if r.check == check and not r.passed and r.value is not None
           and r.limit is not None]
    return max(bad, key=lambda r: abs(r.value - r.limit), default=None)


def _fmt(check: str, value: float) -> str:
    label, scale, unit = PROGRESS_LABELS.get(check, ("value", 1, ""))
    if unit in ("cm", "%", "°"):
        return f"{value * scale:.1f}{'' if unit == '°' else ' '}{unit}".replace(" %", "%")
    return f"{value * scale:.3g} {unit}".strip()


def progress_lines(previous: list[CheckResult], current: list[CheckResult]) -> list[str]:
    """For each check failing now and last round: did the last change help?"""
    lines = []
    for check in dict.fromkeys(r.check for r in current if not r.passed):
        now, before = _worst(current, check), _worst(previous, check)
        if now is None or before is None:
            continue
        label = PROGRESS_LABELS.get(check, ("value",))[0]
        gap_before, gap_now = abs(before.value - before.limit), abs(now.value - now.limit)
        same_text = _fmt(check, before.value) == _fmt(check, now.value)
        if same_text or abs(gap_now - gap_before) <= SAME_WITHIN * max(gap_before, 1e-9):
            verdict = "your last change didn't affect this"
        elif gap_now > gap_before:
            verdict = "your last change made this worse"
        else:
            verdict = "your last change made this better, but it still fails"
        lines.append(f"- [{check}] {label} went from {_fmt(check, before.value)} to "
                     f"{_fmt(check, now.value)}: {verdict}.")
    return lines


def _parts(spec: RobotSpec) -> dict[str, dict[str, Any]]:
    """Spec flattened to named parts: {"joint shoulder": {...}, "link upper_arm": {...}, ...}."""
    parts: dict[str, dict[str, Any]] = {"design": {"control": spec.control}}
    for m in spec.modules:
        d = m.model_dump(exclude_none=True, exclude={"name", "kind"})
        if isinstance(m, ArmChain):
            d.pop("segments")
            for seg in m.segments:
                parts[f"joint {seg.joint.name}"] = seg.joint.model_dump(exclude_none=True, exclude={"name"})
                parts[f"link {seg.link.name}"] = seg.link.model_dump(exclude_none=True, exclude={"name"})
        parts[f"{m.kind} {m.name}"] = d
    return parts


def describe_changes(old: RobotSpec, new: RobotSpec, limit: int = 8) -> str:
    """One line: what changed between two specs, part by part."""
    a, b = _parts(old), _parts(new)
    items = [f"removed {k}" for k in a if k not in b]
    for k in b:
        if k not in a:
            v = b[k]
            detail = ", ".join(f"{f} {v[f]}" for f in ("axis", "motor_id", "length_m", "size_m") if f in v)
            items.append(f"added {k}" + (f" ({detail})" if detail else ""))
            continue
        for f in sorted(set(a[k]) | set(b[k])):
            if a[k].get(f) != b[k].get(f):
                items.append(f"{k} {f} {a[k].get(f)} -> {b[k].get(f)}")
    if not items:
        return "nothing changed"
    more = len(items) - limit
    return "; ".join(items[:limit]) + (f"; and {more} more" if more > 0 else "")


def history_lines(rounds: list[Round]) -> list[str]:
    lines = []
    for i, rnd in enumerate(rounds):
        if i:
            lines.append(f"- Round {i} -> {rnd.number} changes: {describe_changes(rounds[i - 1].spec, rnd.spec)}")
        fails = []
        for check in dict.fromkeys(r.check for r in rnd.failures):
            w = _worst(rnd.results, check)
            fails.append(f"{check} ({PROGRESS_LABELS.get(check, ('value',))[0]} {_fmt(check, w.value)})"
                         if w is not None else check)
        lines.append(f"- Round {rnd.number}: " + (f"failed {', '.join(fails)}" if fails else "passed"))
    return lines


# --------------------------------------------------------------------------- going in circles

SAME_DESIGN_WITHIN = 0.01  # every number within 1%
REPEATS_BEFORE_STOP = 2
TORQUE_CHECKS = {"static_torque", "torque_at_speed"}
CONFLICT_NOTE = (
    "These failures pull against each other: reach needs the arm this long, torque wants it shorter "
    "or lighter. Keep the length reach needs and fix torque with stronger motors near the base "
    "(shoulder, elbow) and lighter outer links (slim PLA, the lightest motors towards the wrist).")


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= SAME_DESIGN_WITHIN * max(abs(a), abs(b)) + 1e-12
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    return a == b


def same_design(a: RobotSpec, b: RobotSpec) -> bool:
    """Equal apart from name and description, every number within 1%."""
    skip = {"name", "description"}
    return _close(a.model_dump(exclude=skip), b.model_dump(exclude=skip))


def earlier_twin(rounds: list[Round]) -> Round | None:
    """The earliest round whose design matches the last round's."""
    last = rounds[-1]
    return next((r for r in rounds[:-1] if same_design(r.spec, last.spec)), None)


def _failed_checks(rnd: Round) -> str:
    return ", ".join(dict.fromkeys(f"{r.check}" + (f" ({r.cause})" if r.cause else "") for r in rnd.failures))


_CLAIMS = {
    "motors": re.compile(r"\b(motors?|servos?|actuators?)\b"),
    "links": re.compile(r"\b(links?|lengthen\w*|shorten\w*|longer|shorter|extend\w*|arm length)\b"),
    "materials": re.compile(r"\b(pla|alumin(?:i)?um|steel|materials?)\b"),
}


def _sentences(text: str) -> list[str]:
    return [t.strip() for t in re.split(r"(?<=[.!?])\s+", text or "") if t.strip()]


def _motors(spec: RobotSpec) -> dict[str, str]:
    return {k: v["motor_id"] for k, v in _parts(spec).items() if "motor_id" in v} | \
        {k: v["wheel_motor_id"] for k, v in _parts(spec).items() if "wheel_motor_id" in v}


def false_claims(old: RobotSpec, new: RobotSpec) -> list[str]:
    """Sentences new in the description that claim motor, link or material changes the spec
    doesn't have, each as a message for the model."""
    said = [t for t in _sentences(new.description) if t not in _sentences(old.description)]
    a, b = _parts(old), _parts(new)
    links_a = {k: {f: v for f, v in d.items() if f != "material"} for k, d in a.items() if k.startswith("link ")}
    links_b = {k: {f: v for f, v in d.items() if f != "material"} for k, d in b.items() if k.startswith("link ")}
    changed = {
        "motors": _motors(old) != _motors(new),
        "links": links_a != links_b,
        "materials": ({k: d.get("material") for k, d in a.items()} != {k: d.get("material") for k, d in b.items()}),
    }
    out = []
    for kind, pattern in _CLAIMS.items():
        claim = next((t for t in said if pattern.search(t.lower())), None)
        if claim is None or changed[kind]:
            continue
        if kind == "motors":
            ids = sorted(set(_motors(new).values()))
            now = (f"every joint still uses {ids[0]}" if len(ids) == 1 else
                   "the motors are unchanged: " + ", ".join(f"{k.split(' ', 1)[1]} {v}" for k, v in _motors(new).items()))
            out.append(f'You said "{claim}", but no motor changed: {now}.')
        elif kind == "links":
            out.append(f'You said "{claim}", but every link has the same shape and size as before.')
        else:
            mats = sorted({d["material"] for k, d in b.items() if k.startswith("link ") and "material" in d})
            now = f"every link is still {mats[0]}" if len(mats) == 1 else "no link changed material"
            out.append(f'You said "{claim}", but {now}.')
    return out


def repair_prompt(spec: RobotSpec, req: Requirements, failures: list[CheckResult],
                  rounds: list[Round] | None = None, note: str | None = None) -> str:
    """`rounds`: every round so far, for progress, history, repeats and claims. `spec` and
    `failures` are the design being repaired (the best round's, after a worse round).
    `note` leads the failure list (e.g. that the loop went back to the best design)."""
    text = (f"{note}\n" if note else "") + format_failures(failures)
    if failures and all(r.check in REACH_CHECKS or r.cause == "unreachable" for r in failures):
        text += "\n" + MOTOR_NOTE
    elif (any(r.check in REACH_CHECKS for r in failures)
          and any(r.check in TORQUE_CHECKS and r.cause == "saturated" for r in failures)):
        text += "\n" + CONFLICT_NOTE
    rounds = rounds or []
    if len(rounds) >= 2:
        twin = earlier_twin(rounds)
        if twin is not None:
            text += (f"\nThis design is the same as round {twin.number}, which failed with: "
                     f"{_failed_checks(twin)}. Try a different change.")
        claims = false_claims(rounds[-2].spec, rounds[-1].spec)
        if claims:
            text += "\n" + "\n".join(claims)
    if len(rounds) >= 2:
        progress = progress_lines(rounds[-2].results, rounds[-1].results)
        if progress:
            text += "\nCompared with the round before:\n" + "\n".join(progress)
    if len(rounds) >= 2:
        text += "\nHistory:\n" + "\n".join(history_lines(rounds))
    return (
        load_prompt("repair")
        .replace("{failures}", text)
        .replace("{spec}", spec.model_dump_json(exclude_none=True))
        .replace("{requirements}", req.model_dump_json())
    )


def repair_loop(
    client: LLMClient,
    req: Requirements,
    catalog: dict[str, Motor] | None = None,
    initial_spec: RobotSpec | None = None,
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    simulate: bool = True,
    on_event: Event | None = None,
) -> RepairOutcome:
    """Round 1 checks the initial design (asked of the LLM unless given). Each later
    round sends the failed checks back and checks the revised spec. With
    `simulate`, the sim tests run in any round whose static checks all pass.
    `on_event` gets one line of text per step and per check result, as they happen."""
    emit = on_event or (lambda line: None)
    catalog = catalog if catalog is not None else load_motor_catalog()
    outcome = RepairOutcome(requirements=req, max_rounds=max_rounds)
    if initial_spec is None:
        emit("Designing...")
    spec = initial_spec if initial_spec is not None else design_robot(client, req, catalog)
    system = design_system_prompt(catalog)  # schema + catalog, same as the design step
    repeats = 0

    for number in range(1, max_rounds + 1):
        emit(f"Round {number}: {spec.name}: static checks")
        results, runs = run_checks(spec, req, catalog), []
        for r in results:
            emit(_line(r))
        if simulate and all(r.passed for r in results):
            for test in sim_tests_for(spec, req):
                emit(f"  ...simulating {test.__name__}")
                run = test(spec, req, catalog)
                runs.append(run)
                emit(_line(run.result))
            results = results + [run.result for run in runs]
        elif simulate:
            emit("  static checks failed, so simulation is skipped this round")
        rnd = Round(number, spec, results, runs)
        outcome.rounds.append(rnd)
        emit(f"Round {number}: " + ("all checks pass" if rnd.passed else f"{len(rnd.failures)} failed"))
        if len(outcome.rounds) >= 2:
            for line in progress_lines(outcome.rounds[-2].results, rnd.results):
                emit("  " + line[2:])
        infeasible = [r.message for r in rnd.failures if r.cause == "infeasible"]
        if infeasible:
            outcome.infeasible = infeasible
            outcome.stop_reason = "the requirements are beyond the motor catalog, so no design can pass"
            emit(f"Stopping: {outcome.stop_reason}.")
            break
        twin = earlier_twin(outcome.rounds) if len(outcome.rounds) >= 2 else None
        if twin is not None:
            repeats += 1
            emit(f"  This design is the same as round {twin.number}.")
            if repeats >= REPEATS_BEFORE_STOP:
                outcome.stop_reason = (f"repeating designs: round {number} is the same as round {twin.number} "
                                       f"({repeats} repeats)")
                emit(f"Stopping: {outcome.stop_reason}.")
                break
        if rnd.passed or number == max_rounds:
            break
        start, note = rnd, None
        best = best_round(outcome.rounds)
        if round_rank(rnd) > round_rank(best):
            start = best
            note = (f"Round {rnd.number} was worse than round {best.number} ({failed_checks(rnd)} failures vs "
                    f"{failed_checks(best)}). Starting again from round {best.number}'s design.")
            emit("  " + note)
        emit(f"Asking the model to repair {len(start.failures)} failure(s)...")
        try:
            spec = client.complete_json(
                system, repair_prompt(start.spec, req, start.failures, outcome.rounds, note), RobotSpec,
                context={"motor_ids": set(supported_motors(catalog))},
            )
        except LLMOutputError as exc:  # the model's fault; network/rate-limit errors propagate
            outcome.stop_reason = f"repair request after round {number} failed: {exc}"
            break
    return outcome
