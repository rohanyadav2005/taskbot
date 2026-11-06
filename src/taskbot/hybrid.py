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
from taskbot.llm import LLMClient, LLMError
from taskbot.repair import RepairOutcome, Round, _line, format_failures, sim_tests_for
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec
from taskbot.sizing import SizedDesign, SizingFailure, size_design
from taskbot.timing import timed

MAX_INTENT_REVISIONS = 3


@dataclass
class HybridOutcome(RepairOutcome):
    design_mode: str = "hybrid"
    intents: list[DesignIntent] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)          # sizing and autofix, in order
    provenance: dict[str, str] = field(default_factory=dict)
    approximate: str | None = None                          # requirement relaxed (--nearest), or why
    approximate_kind: str = ""                              # "requirements": they do less than the task asks
    infeasible_options: list[str] = field(default_factory=list)
    infeasible_kind: str = ""                               # "precision": the tolerance is beyond the catalog
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


def _intent(client, req, feedback, fall_back) -> DesignIntent | None:
    """The LLM's intent; the default intent (no LLM needed) if its replies can't be used.
    None for a revision that fails: there is nothing better to try."""
    from taskbot.normalize import default_intent

    try:
        return request_intent(client, req, feedback)
    except LLMError as exc:
        if getattr(exc, "propagate", False):
            raise
        step = "intent_revision" if feedback else "intent"
        if feedback:
            fall_back(step, f"no usable revised intent ({str(exc).splitlines()[0]})")
            return None
        fall_back(step, f"no usable intent ({str(exc).splitlines()[0]}); used the default intent")
        return default_intent(req)


def _alternative(req, tried, fall_back) -> DesignIntent | None:
    """Code's next untried intent, when the LLM gives no usable revision."""
    from taskbot.normalize import alternative_intent

    alt = alternative_intent(req, tried)
    if alt is not None:
        fall_back("intent_revision", "code tried its next untried intent instead: "
                  + alt.model_dump_json(exclude_none=True, exclude={"rationale", "not_modelled"}))
    return alt


def hybrid_loop(client: LLMClient, req: Requirements, catalog: dict[str, Motor], *, simulate: bool = True,
                nearest: bool = False, on_event: Callable[[str], None] | None = None,
                on_fallback: Callable[[str, str], None] | None = None) -> HybridOutcome:
    emit = on_event or (lambda line: None)
    fall_back = on_fallback or (lambda step, what: emit(f"Fallback ({step}): {what}"))
    out = HybridOutcome(requirements=req, max_rounds=0)

    def check(spec: RobotSpec, target: Requirements) -> list[CheckResult]:
        number = len(out.rounds) + 1
        emit(f"Round {number}: {spec.name}: static checks")
        with timed("static_checks", f"round {number}"):
            results, runs = run_checks(spec, target, catalog), []
        for r in results:
            emit(_line(r))
        if simulate and all(r.passed for r in results):
            for test in sim_tests_for(spec, target):
                emit(f"  ...simulating {test.__name__}")
                with timed("sim", f"round {number}: {test.__name__}"):
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
    intent = _intent(client, req, None, fall_back)
    for revision in range(MAX_INTENT_REVISIONS + 1):
        out.intents.append(intent)
        emit(f"Intent{f' (revision {revision})' if revision else ''}: " + intent.model_dump_json(exclude_none=True))
        emit("Sizing...")
        try:
            with timed("sizing", f"intent {revision + 1}"):
                sized = size_design(intent, req, catalog, nearest=nearest)
        except Exception as exc:  # a sizer bug must not end the run without a report
            import traceback

            sized = SizingFailure("checks", f"the sizer failed: {type(exc).__name__}: {exc}",
                                  steps=[traceback.format_exc(limit=4)])
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
        if sized.spec is None:  # nothing to check: try another intent, if any revisions are left
            out.stop_reason = sized.reason
            emit(f"Sizing failed: {sized.reason}")
            if revision == MAX_INTENT_REVISIONS:
                break
            revised = _intent(client, req, f"Sizing your last intent {intent.model_dump_json(exclude_none=True)} "
                              f"failed: {sized.reason}\nChoose a different intent; don't give any numbers.", fall_back)
            revised = revised or _alternative(req, out.intents, fall_back)
            if revised is None:
                break
            intent = revised
            continue
        target = req
        if isinstance(sized, SizedDesign) and sized.approximate:
            out.approximate, out.sized_for, target = sized.approximate, sized.requirements, sized.requirements
            emit(f"APPROXIMATE: relaxed {sized.approximate}")
        spec = sized.spec
        out.sized_spec = spec
        results = check(spec, target)
        with timed("autofix", f"intent {revision + 1}"):
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
        revised = _intent(client, req, revision_message(intent, failures), fall_back) or \
            _alternative(req, out.intents, fall_back)
        if revised is None:
            out.stop_reason = "the model gave no usable revised intent, and code had no untried intent left"
            break
        intent = revised
    _precision_limit(out, req)
    return out


SIM_CHECKS = ("reach_targets", "carry_payload", "hold_still")


def _precision_limit(out: HybridOutcome, req: Requirements) -> None:
    """A tight tolerance (the sizer chose the stiffest motors that carry the arm)
    that every design still misses only through compliance in the sims: the catalog's position
    loops can't hold the tool that precisely. INFEASIBLE, with the best precision any design
    reached (SimRun.achieved_tolerance_m)."""
    from taskbot.sizing import TIGHT_TOLERANCE_M

    tol = req.position_tolerance_m
    if out.passed or out.infeasible or not out.rounds or tol > TIGHT_TOLERANCE_M:
        return
    fails = out.final.failures
    if not fails or any(r.cause != "compliance" or r.check not in SIM_CHECKS for r in fails):
        return
    reached = [(max(a), rnd) for rnd in out.rounds
               if (a := [run.achieved_tolerance_m for run in rnd.sim_runs if run.achieved_tolerance_m is not None])]
    if not reached:
        return
    best, rnd = min(reached, key=lambda t: t[0])
    if best <= tol:  # missed on settling time, not precision
        return
    mm = best * 1000
    out.infeasible = [
        f"No design TaskBot sized from this catalog holds the tool within the task's {tol * 1000:.3g} mm: the most "
        f"precise one (round {rnd.number}, control \"{rnd.spec.control}\", the stiffest catalog motors that carry the "
        f"arm) holds about {mm:.2g} mm (its worst final error, path deviation / 3 and overshoot / 2 over the sim "
        f"tests). The catalog's position loops are too soft for {tol * 1000:.3g} mm at this reach and payload."]
    out.infeasible_options = [
        f"A position tolerance of about {mm:.2g} mm, if the task allows it, with the round {rnd.number} design.",
        "A shorter reach or a lighter payload (less sag at every joint), or motors stiffer than any in the catalog."]
    out.infeasible_kind = "precision"
    out.stop_reason = f"the catalog's motors can't hold the tool to {tol * 1000:.3g} mm (best about {mm:.2g} mm)"
