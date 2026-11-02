"""The "hybrid" design mode (the default): the LLM chooses, code sizes.

    requirements -> DesignIntent (LLM) -> size_design (code) -> checks + sims
        -> autofix (code, up to 5 rounds) -> still failing? -> revised DesignIntent (LLM,
           at most MAX_INTENT_REVISIONS) -> size again ...

The LLM never writes a number. It is asked again only when automatic fixes run out, with the
remaining failures, for a different layout, drive or mount. The "llm" mode (repair.repair_loop,
the LLM writes the whole spec) is kept unchanged for comparison.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from taskbot.autofix import autofix_loop
from taskbot.checks import CheckResult, run_checks
from taskbot.intent import DesignIntent, request_intent
from taskbot.llm import LLMClient
from taskbot.repair import RepairOutcome, Round, _line, format_failures, sim_tests_for
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec
from taskbot.sizing import SizedDesign, SizingFailure, size_design

MAX_INTENT_REVISIONS = 3


@dataclass
class HybridOutcome(RepairOutcome):
    design_mode: str = "hybrid"
    intents: list[DesignIntent] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)          # sizing and autofix, in order
    provenance: dict[str, str] = field(default_factory=dict)
    approximate: str | None = None                          # requirement relaxed (--nearest)
    infeasible_options: list[str] = field(default_factory=list)
    sized_for: Requirements | None = None                   # the relaxed requirements, if approximate
    sized_spec: RobotSpec | None = None                     # what the sizer made from the last intent
    prompt: str = ""                                        # the user's words, for the report

    @property
    def passed(self) -> bool:  # an approximate design never counts as a pass
        return super().passed and self.approximate is None


def revision_message(intent: DesignIntent, failures: list[CheckResult]) -> str:
    return ("The design sized from your last intent " + intent.model_dump_json(exclude_none=True)
            + " still fails after automatic fixes:\n" + format_failures(failures)
            + "\nChoose a different intent that avoids these failures (another arm layout, drive or "
              "arm mount). Code will size it again; don't give any numbers.")


def hybrid_loop(client: LLMClient, req: Requirements, catalog: dict[str, Motor], *, simulate: bool = True,
                nearest: bool = False, on_event: Callable[[str], None] | None = None) -> HybridOutcome:
    emit = on_event or (lambda line: None)
    out = HybridOutcome(requirements=req, max_rounds=0)

    def check(spec: RobotSpec, target: Requirements) -> list[CheckResult]:
        number = len(out.rounds) + 1
        emit(f"Round {number}: {spec.name}: static checks")
        results, runs = run_checks(spec, target, catalog), []
        for r in results:
            emit(_line(r))
        if simulate and all(r.passed for r in results):
            for test in sim_tests_for(spec, target):
                emit(f"  ...simulating {test.__name__}")
                run = test(spec, target, catalog)
                runs.append(run)
                emit(_line(run.result))
            results = results + [run.result for run in runs]
        out.rounds.append(Round(number, spec, results, runs))
        out.max_rounds = len(out.rounds)
        rnd = out.rounds[-1]
        emit(f"Round {number}: " + ("all checks pass" if rnd.passed else f"{len(rnd.failures)} failed"))
        return results

    emit("Choosing the design intent...")
    intent = request_intent(client, req)
    for revision in range(MAX_INTENT_REVISIONS + 1):
        out.intents.append(intent)
        emit(f"Intent{f' (revision {revision})' if revision else ''}: " + intent.model_dump_json(exclude_none=True))
        emit("Sizing...")
        sized = size_design(intent, req, catalog, nearest=nearest)
        steps = sized.steps
        out.steps += [f"sizing{f' (intent {revision + 1})' if revision else ''}: {s}" for s in steps]
        for s in steps:
            emit(f"  sizer: {s}")
        out.provenance = sized.provenance
        if isinstance(sized, SizingFailure) and sized.kind == "infeasible":
            out.infeasible = [sized.reason]
            out.infeasible_options = sized.options
            out.stop_reason = "the sized design can't carry the load with any catalog motor"
            if sized.spec is not None:
                check(sized.spec, req)
            emit(f"Stopping: {sized.reason}")
            break
        target = req
        if isinstance(sized, SizedDesign) and sized.approximate:
            out.approximate, out.sized_for, target = sized.approximate, sized.requirements, sized.requirements
            emit(f"APPROXIMATE: relaxed {sized.approximate}")
        spec = sized.spec
        out.sized_spec = spec
        results = check(spec, target)
        fixed = autofix_loop(spec, target, catalog, lambda s: check(s, target), results=results, on_event=emit)
        out.steps += fixed.steps
        failures = [r for r in fixed.results if not r.passed]
        if not failures:
            break
        if any(r.cause == "infeasible" for r in failures):
            out.infeasible = [r.message for r in failures if r.cause == "infeasible"]
            out.stop_reason = "the requirements are beyond the motor catalog, so no design can pass"
            break
        if revision == MAX_INTENT_REVISIONS:
            out.stop_reason = (f"automatic fixes and {MAX_INTENT_REVISIONS} intent revisions didn't fix it "
                               f"({fixed.stop})")
            break
        emit(f"Autofix stopped ({fixed.stop}); asking the model for a different intent...")
        intent = request_intent(client, req, revision_message(intent, failures))
    return out
