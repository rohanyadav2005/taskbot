"""design -> build -> check -> repair, until every check passes or rounds run out."""

from __future__ import annotations

from dataclasses import dataclass, field

from taskbot.checks import CheckResult, run_checks
from taskbot.design import design_robot, design_system_prompt
from taskbot.llm import LLMClient, LLMError
from taskbot.prompts import load_prompt
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec, load_motor_catalog

DEFAULT_MAX_ROUNDS = 5


@dataclass(frozen=True)
class Round:
    number: int
    spec: RobotSpec
    results: list[CheckResult]

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
    return "\n".join(f"- [{r.check}] {r.message}" for r in failures)


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
) -> RepairOutcome:
    """Round 1 checks the initial design (asked of the LLM unless given). Each later
    round sends the failed checks back and checks the revised spec."""
    catalog = catalog if catalog is not None else load_motor_catalog()
    outcome = RepairOutcome(requirements=req, max_rounds=max_rounds)
    spec = initial_spec if initial_spec is not None else design_robot(client, req, catalog)
    system = design_system_prompt(catalog)  # schema + catalog, same as the design step

    for number in range(1, max_rounds + 1):
        rnd = Round(number, spec, run_checks(spec, req, catalog))
        outcome.rounds.append(rnd)
        if rnd.passed or number == max_rounds:
            break
        try:
            spec = client.complete_json(
                system, repair_prompt(spec, req, rnd.failures), RobotSpec,
                context={"motor_ids": set(catalog)},
            )
        except LLMError as exc:
            outcome.stop_reason = f"repair request after round {number} failed: {exc}"
            break
    return outcome
