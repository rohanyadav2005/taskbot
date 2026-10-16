"""design -> build -> check -> simulate -> repair, until everything passes or rounds run out.

Each round runs the static checks first; the (slower) simulation tests run only
once every static check passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from taskbot.checks import CheckResult, run_checks
from taskbot.design import design_robot, design_system_prompt
from taskbot.llm import LLMClient, LLMError
from taskbot.prompts import load_prompt
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec, load_motor_catalog, supported_motors
from taskbot.sim_tests import SIM_TESTS, SimRun, run_sim_tests

DEFAULT_MAX_ROUNDS = 5


SIM_TEST_NAMES = [t.__name__ for t in SIM_TESTS]


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
) -> RepairOutcome:
    """Round 1 checks the initial design (asked of the LLM unless given). Each later
    round sends the failed checks back and checks the revised spec. With
    `simulate`, the sim tests run in any round whose static checks all pass."""
    catalog = catalog if catalog is not None else load_motor_catalog()
    outcome = RepairOutcome(requirements=req, max_rounds=max_rounds)
    spec = initial_spec if initial_spec is not None else design_robot(client, req, catalog)
    system = design_system_prompt(catalog)  # schema + catalog, same as the design step

    for number in range(1, max_rounds + 1):
        results, runs = run_checks(spec, req, catalog), []
        if simulate and all(r.passed for r in results):
            runs = run_sim_tests(spec, req, catalog)
            results = results + [run.result for run in runs]
        rnd = Round(number, spec, results, runs)
        outcome.rounds.append(rnd)
        if rnd.passed or number == max_rounds:
            break
        try:
            spec = client.complete_json(
                system, repair_prompt(spec, req, rnd.failures), RobotSpec,
                context={"motor_ids": set(supported_motors(catalog))},
            )
        except LLMError as exc:
            outcome.stop_reason = f"repair request after round {number} failed: {exc}"
            break
    return outcome
