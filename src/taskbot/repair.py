"""design -> build -> check -> simulate -> repair, until everything passes or rounds run out.

Each round runs the static checks first; the (slower) simulation tests run only
once every static check passes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from taskbot.checks import CheckResult, run_checks
from taskbot.design import design_robot, design_system_prompt
from taskbot.llm import LLMClient, LLMError
from taskbot.prompts import load_prompt
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec, load_motor_catalog, supported_motors
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


def repair_prompt(spec: RobotSpec, req: Requirements, failures: list[CheckResult]) -> str:
    return (
        load_prompt("repair")
        .replace("{failures}", format_failures(failures))
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
        if rnd.passed or number == max_rounds:
            break
        emit(f"Asking the model to repair {len(rnd.failures)} failure(s)...")
        try:
            spec = client.complete_json(
                system, repair_prompt(spec, req, rnd.failures), RobotSpec,
                context={"motor_ids": set(supported_motors(catalog))},
            )
        except LLMError as exc:
            outcome.stop_reason = f"repair request after round {number} failed: {exc}"
            break
    return outcome
