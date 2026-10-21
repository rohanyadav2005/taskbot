"""design -> build -> check -> simulate -> repair, until everything passes or rounds run out.

Each round runs the static checks first; the (slower) simulation tests run only
once every static check passes.

Repair requests carry, besides the failures: for each check that also failed last
round, whether the last change made it better or worse; a compact history of every
round (what failed, what changed); and, when only reach-type checks fail, a note that
motors can't fix them.
"""

from __future__ import annotations

from collections.abc import Callable
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
    stop_reason: str | None = None  # set when the loop ended early on an LLM error

    @property
    def passed(self) -> bool:
        return bool(self.rounds) and self.rounds[-1].passed

    @property
    def final(self) -> Round:
        return self.rounds[-1]


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


def repair_prompt(spec: RobotSpec, req: Requirements, failures: list[CheckResult],
                  rounds: list[Round] | None = None) -> str:
    """`rounds`: every round so far (the last one produced `failures`), for progress and history."""
    text = format_failures(failures)
    if failures and all(r.check in REACH_CHECKS or r.cause == "unreachable" for r in failures):
        text += "\n" + MOTOR_NOTE
    rounds = rounds or []
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
        if rnd.passed or number == max_rounds:
            break
        emit(f"Asking the model to repair {len(rnd.failures)} failure(s)...")
        try:
            spec = client.complete_json(
                system, repair_prompt(spec, req, rnd.failures, outcome.rounds), RobotSpec,
                context={"motor_ids": set(supported_motors(catalog))},
            )
        except LLMOutputError as exc:  # the model's fault; network/rate-limit errors propagate
            outcome.stop_reason = f"repair request after round {number} failed: {exc}"
            break
    return outcome
