"""Evaluation benchmark: run many prompts through the full pipeline and score them.

    python bench/run.py --backend fake --limit 5
    python bench/run.py --backend openai_compat --split all
    python bench/run.py --compare bench/results/a.csv bench/results/b.csv

Prompts live in bench/prompts.yaml (main) and bench/holdout.yaml (reported separately).
Each run keeps its state in bench/results/<name>/results.jsonl, one line per finished
prompt, so an interrupted run (rate limits, crash, Ctrl-C) continues where it stopped
when the same command (same --name) is run again. The report is
bench/results/<name>.md and .csv, where <name> defaults to <date>_<backend>_<model>.

A prompt "passes" when an in-scope task ends with a design that passes every check and
sim test, or an out-of-scope task is rejected without a design. Expectations about the
extracted requirements (robot type, reach, payload, tolerance) are scored separately.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from taskbot.llm import LLMClient, LLMError, LLMOutputError, LLMRateLimitError
from taskbot.requirements import Requirements

CATEGORIES = ("easy_arm", "hard_arm", "mobile", "mobile_manipulator", "vague", "out_of_scope")
Category = Literal["easy_arm", "hard_arm", "mobile", "mobile_manipulator", "vague", "out_of_scope"]
RobotType = Literal["arm", "mobile", "mobile_manipulator"]
SPLITS = ("main", "holdout")

FAKE_WARNING = ("FAKE BACKEND — scripted responses. Tests the benchmark itself only. "
                "Not a measure of design ability.")

# Record statuses. FINAL ones are never re-run on resume; the others are.
DONE, NO_FAKE, ERROR, RATE_LIMITED = "done", "no_fake_response", "error", "rate_limited"
FINAL = {DONE, NO_FAKE}


# --------------------------------------------------------------------------- prompts


Range = tuple[float, float]


class Expectations(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_type: RobotType | list[RobotType] | None = None
    should_reject: bool = False
    min_reach_m: float | None = Field(default=None, gt=0)
    payload_kg: Range | None = None
    base_payload_kg: Range | None = None
    tolerance_m: Range | None = None
    terrain: Literal["flat", "carpet", "outdoor"] | None = None

    @field_validator("payload_kg", "base_payload_kg", "tolerance_m")
    @classmethod
    def _range(cls, v: Range | None) -> Range | None:
        if v is not None and not 0 <= v[0] <= v[1]:
            raise ValueError(f"range must be [min, max] with 0 <= min <= max, got {list(v)}")
        return v

    @model_validator(mode="after")
    def _reject_alone(self) -> Expectations:
        others = [k for k in type(self).model_fields if k != "should_reject" and getattr(self, k) is not None]
        if self.should_reject and others:
            raise ValueError(f"should_reject: true can't be combined with {others}")
        return self

    @property
    def robot_types(self) -> list[str] | None:
        if self.robot_type is None:
            return None
        return [self.robot_type] if isinstance(self.robot_type, str) else list(self.robot_type)


class BenchPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9_]+$")
    category: Category
    prompt: str = Field(min_length=3)
    expect: Expectations = Field(default_factory=Expectations)
    note: str | None = None
    split: Literal["main", "holdout"] = "main"  # set by the loader from the file

    @model_validator(mode="after")
    def _out_of_scope_rejects(self) -> BenchPrompt:
        if (self.category == "out_of_scope") != self.expect.should_reject:
            raise ValueError("out_of_scope prompts need should_reject: true, and only they may have it")
        return self


class PromptFileError(ValueError):
    pass


def load_prompts(path: Path, split: str = "main") -> list[BenchPrompt]:
    """Parse and validate one prompts file. Errors name the file and the prompt."""
    try:
        data = yaml.safe_load(Path(path).read_text("utf-8"))
    except yaml.YAMLError as exc:
        raise PromptFileError(f"{path}: not valid YAML: {exc}") from exc
    if not isinstance(data, list):
        raise PromptFileError(f"{path}: expected a list of prompts, got {type(data).__name__}")
    prompts, errors = [], []
    for i, item in enumerate(data):
        label = item.get("id", f"#{i + 1}") if isinstance(item, dict) else f"#{i + 1}"
        if isinstance(item, dict) and "split" in item:
            errors.append(f"{label}: 'split' comes from the file, don't set it")
            continue
        try:
            prompts.append(BenchPrompt.model_validate({**item, "split": split} if isinstance(item, dict) else item))
        except ValidationError as exc:
            errors.append(f"{label}: " + "; ".join(
                f"{'.'.join(map(str, e['loc'])) or '(root)'}: {e['msg']}" for e in exc.errors(include_url=False)))
    dupes = [k for k, n in Counter(p.id for p in prompts).items() if n > 1]
    errors += [f"duplicate id '{d}'" for d in dupes]
    if errors:
        raise PromptFileError(f"{path}:\n" + "\n".join(f"  {e}" for e in errors))
    return prompts


def load_suite(bench_dir: Path, split: str = "main") -> list[BenchPrompt]:
    """Main and/or holdout prompts. Ids and prompt texts must be unique across both files."""
    bench_dir = Path(bench_dir)
    main = load_prompts(bench_dir / "prompts.yaml", "main")
    holdout = load_prompts(bench_dir / "holdout.yaml", "holdout")
    both = main + holdout
    for attr in ("id", "prompt"):
        dupes = [k for k, n in Counter(" ".join(getattr(p, attr).lower().split()) for p in both).items() if n > 1]
        if dupes:
            raise PromptFileError(f"{attr} used in both prompts.yaml and holdout.yaml: {dupes}")
    return {"main": main, "holdout": holdout, "all": both}[split]


def select(prompts: list[BenchPrompt], categories: list[str] | None = None,
           limit: int | None = None) -> list[BenchPrompt]:
    if categories:
        unknown = set(categories) - set(CATEGORIES)
        if unknown:
            raise ValueError(f"unknown categories {sorted(unknown)}; choose from {list(CATEGORIES)}")
        prompts = [p for p in prompts if p.category in categories]
    return prompts[:limit] if limit else prompts


# --------------------------------------------------------------------------- scoring


def score_expectations(req: Requirements, exp: Expectations) -> tuple[bool | None, list[str]]:
    """(robot type correct or None if not specified, list of unmet expectations)."""
    misses = []
    types = exp.robot_types
    type_ok = None if types is None else req.robot_type in types
    if type_ok is False:
        misses.append(f"robot_type {req.robot_type}, expected {' or '.join(types)}")

    def in_range(name: str, value: float | None, rng: Range | None, unit: str):
        if rng is None:
            return
        if value is None or not rng[0] <= value <= rng[1]:
            misses.append(f"{name} {value if value is not None else 'missing'}{'' if value is None else ' ' + unit}, "
                          f"expected {rng[0]:g}–{rng[1]:g} {unit}")

    if exp.min_reach_m is not None and (req.reach_m is None or req.reach_m < exp.min_reach_m):
        misses.append(f"reach_m {req.reach_m}, expected at least {exp.min_reach_m:g} m")
    in_range("payload_kg", req.payload_kg, exp.payload_kg, "kg")
    in_range("base_payload_kg", req.base_payload_kg, exp.base_payload_kg, "kg")
    in_range("position_tolerance_m", req.position_tolerance_m, exp.tolerance_m, "m")
    if exp.terrain is not None and req.terrain != exp.terrain:
        misses.append(f"terrain {req.terrain}, expected {exp.terrain}")
    return type_ok, misses


@dataclass
class Record:
    """One prompt's result: a line of results.jsonl and a row of the CSV."""

    id: str
    category: str
    split: str
    prompt: str
    status: str
    should_reject: bool
    passed: bool = False
    rejected: bool | None = None
    rejection_correct: bool | None = None
    requirements_valid: bool | None = None
    valid_spec: bool | None = None
    robot_type: str | None = None
    robot_type_correct: bool | None = None
    expectations_met: bool | None = None
    expectation_failures: list[str] = field(default_factory=list)
    rounds: int = 0
    final_checks: list[dict] = field(default_factory=list)   # {check, passed, cause}
    failed_any_round: list[dict] = field(default_factory=list)  # {round, check, cause}
    unfixed: list[dict] = field(default_factory=list)        # {check, cause, message}
    stop_reason: str | None = None
    error: str | None = None
    tokens_in: int = 0
    tokens_out: int = 0
    json_retries: int = 0
    rate_limit_waits: int = 0
    time_s: float = 0.0
    report: str | None = None   # relative to the run directory
    backend: str = ""
    model: str = ""
    provider: str = ""

    @property
    def scored(self) -> bool:
        return self.status == DONE


@dataclass
class RunMeta:
    backend: str
    model: str
    provider: str
    name: str
    started: str = ""

    @property
    def is_fake(self) -> bool:
        return self.backend == "fake"


def run_one(p: BenchPrompt, client: LLMClient, run_dir: Path, catalog, meta: RunMeta, *,
            max_rounds: int, simulate: bool, record: bool) -> Record:
    from taskbot.fake import NoFakeResponse
    from taskbot.pipeline import run_pipeline

    out_root = run_dir / "out" / p.id
    if out_root.exists():
        shutil.rmtree(out_root)  # a re-run (after an error) starts clean
    out_root.mkdir(parents=True)
    rec = Record(p.id, p.category, p.split, p.prompt, DONE, p.expect.should_reject,
                 backend=meta.backend, model=meta.model, provider=meta.provider)
    start = time.monotonic()
    try:
        result = run_pipeline(client, p.prompt, catalog, out_root, max_rounds=max_rounds,
                              simulate=simulate, record=record)
    except NoFakeResponse as exc:
        rec.status, rec.error = NO_FAKE, str(exc)
    except LLMRateLimitError as exc:
        rec.status, rec.error = RATE_LIMITED, str(exc)
    except LLMOutputError as exc:  # the model's fault: scored as a failure
        rec.requirements_valid = exc.schema_name != "Requirements"
        rec.valid_spec = False if exc.schema_name == "RobotSpec" else None
        rec.rejected = False
        rec.rejection_correct = not p.expect.should_reject if p.expect.should_reject else None
        rec.error = str(exc)
        rec.unfixed = [{"check": f"invalid_{exc.schema_name.lower()}", "cause": "invalid_json",
                        "message": exc.last_error}]
    except LLMError as exc:  # network, auth, timeout: not the model's fault; re-run on resume
        rec.status, rec.error = ERROR, str(exc)
    except Exception as exc:  # a bug in TaskBot: record it and carry on with the others
        rec.status = ERROR
        rec.error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=4)}"
    else:
        _score_result(rec, p, result, run_dir)
    u = client.usage
    rec.tokens_in, rec.tokens_out = u.input_tokens, u.output_tokens
    rec.json_retries, rec.rate_limit_waits = u.json_retries, u.rate_limit_waits
    rec.time_s = round(time.monotonic() - start, 2)
    return rec


def _score_result(rec: Record, p: BenchPrompt, result, run_dir: Path) -> None:
    req = result.requirements
    rec.requirements_valid = True
    rec.robot_type = req.robot_type
    rec.rejected = result.rejected
    # Forward slashes on every OS: this goes into Markdown links and the CSV.
    rec.report = (result.out_dir / "report.md").relative_to(run_dir).as_posix()
    type_ok, misses = score_expectations(req, p.expect)
    if p.expect.should_reject:
        rec.rejection_correct = result.rejected
        rec.passed = result.rejected
        if not result.rejected:
            misses.append("should have been rejected as out of scope")
    elif result.rejected:
        misses.append("rejected as out of scope, but TaskBot can build this")
    rec.robot_type_correct = type_ok
    rec.expectation_failures = misses
    rec.expectations_met = not misses
    if result.error is not None:
        rec.valid_spec = False
        rec.error = result.error
        rec.unfixed = [{"check": "invalid_robotspec", "cause": "invalid_json", "message": result.error}]
        return
    o = result.outcome
    if o is None:
        return
    rec.valid_spec = True
    rec.rounds = len(o.rounds)
    rec.passed = o.passed and not p.expect.should_reject
    rec.stop_reason = o.stop_reason
    rec.final_checks = [{"check": r.check, "passed": r.passed, "cause": r.cause} for r in o.final.results]
    rec.failed_any_round = [{"round": rnd.number, "check": r.check, "cause": r.cause}
                            for rnd in o.rounds for r in rnd.failures]
    if not o.passed:
        rec.unfixed = [{"check": r.check, "cause": r.cause or "none", "message": r.message}
                       for r in o.final.failures]
        if o.stop_reason:
            rec.unfixed.append({"check": "repair_request", "cause": "invalid_json", "message": o.stop_reason})


# --------------------------------------------------------------------------- running


def _load_state(path: Path) -> dict[str, Record]:
    state: dict[str, Record] = {}
    if path.exists():
        for line in path.read_text("utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get("report"):  # state written on Windows before reports used forward slashes
                    d["report"] = d["report"].replace("\\", "/")
                state[d["id"]] = Record(**d)  # later lines win
    return state


def run_bench(prompts: list[BenchPrompt], run_dir: Path, client_factory: Callable[[], LLMClient],
              meta: RunMeta, catalog=None, *, concurrency: int = 2, max_rounds: int = 5,
              simulate: bool = True, record: bool = False,
              progress: Callable[[str], None] | None = None) -> list[Record]:
    """Run every prompt not already finished in `run_dir`, `concurrency` at a time.
    Returns the records for `prompts`, in order (missing ones are left out)."""
    from taskbot.schema import load_motor_catalog

    say = progress or (lambda line: None)
    catalog = catalog if catalog is not None else load_motor_catalog()
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text("utf-8"))
        if (old["backend"], old["model"]) != (meta.backend, meta.model):
            raise ValueError(f"{run_dir} holds a run of {old['backend']} / {old['model']}; this is "
                             f"{meta.backend} / {meta.model}. Use a different --name.")
        meta.started = old.get("started", "")
    else:
        meta.started = dt.datetime.now().isoformat(timespec="seconds")
        meta_path.write_text(json.dumps(asdict(meta), indent=2) + "\n", "utf-8")
    state_path = run_dir / "results.jsonl"
    state = _load_state(state_path)
    todo = [p for p in prompts if p.id not in state or state[p.id].status not in FINAL]
    if len(todo) < len(prompts):
        say(f"Resuming: {len(prompts) - len(todo)} of {len(prompts)} prompts already done.")
    lock = threading.Lock()

    def work(p: BenchPrompt) -> Record:
        return run_one(p, client_factory(), run_dir, catalog, meta, max_rounds=max_rounds,
                       simulate=simulate, record=record)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                with state_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")
                state[rec.id] = rec
            say(f"[{n}/{len(todo)}] {rec.id}: {_outcome(rec)} ({rec.time_s:g} s)")
    return [state[p.id] for p in prompts if p.id in state]


def _outcome(r: Record) -> str:
    if r.status != DONE:
        return {NO_FAKE: "no fake response", RATE_LIMITED: "rate-limited (rerun to resume)",
                ERROR: "error (rerun to retry)"}[r.status] + (f": {r.error.splitlines()[0]}" if r.error and r.status == ERROR else "")
    if r.should_reject:
        return "rejected (correct)" if r.rejected else "NOT rejected"
    if r.rejected:
        return "wrongly rejected"
    if r.valid_spec is False:
        return "invalid spec"
    return f"{'PASS' if r.passed else 'FAIL'} in {r.rounds} round{'s' * (r.rounds != 1)}"


# --------------------------------------------------------------------------- reports

CSV_FIELDS = ["id", "split", "category", "status", "passed", "rejected", "should_reject",
              "rejection_correct", "requirements_valid", "valid_spec", "robot_type",
              "robot_type_correct", "expectations_met", "expectation_failures", "rounds",
              "final_failures", "unfixed_count", "tokens_in", "tokens_out", "json_retries",
              "rate_limit_waits", "time_s", "report", "backend", "model", "provider", "error"]


def _yn(v: bool | None) -> str:
    return "" if v is None else ("y" if v else "n")


def write_csv(records: list[Record], path: Path, run_name: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in records:
            row = {k: getattr(r, k) for k in CSV_FIELDS if hasattr(r, k)}
            row |= {k: _yn(getattr(r, k)) for k in ("passed", "rejected", "should_reject", "rejection_correct",
                                                     "requirements_valid", "valid_spec", "robot_type_correct",
                                                     "expectations_met")}
            row["expectation_failures"] = "; ".join(r.expectation_failures)
            row["final_failures"] = "; ".join(f"{c['check']}:{c['cause'] or 'none'}"
                                              for c in r.final_checks if not c["passed"])
            row["unfixed_count"] = len(r.unfixed)
            row["report"] = f"{run_name}/{r.report}" if r.report else ""
            row["error"] = (r.error or "").splitlines()[0] if r.error else ""
            w.writerow(row)


def _pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "–"


def _rate_table(records: list[Record]) -> list[str]:
    lines = ["| Category | Scored | Passed | Robot type correct | Expectations met | Avg rounds to pass |",
             "|---|---|---|---|---|---|"]
    for cat in CATEGORIES:
        rs = [r for r in records if r.category == cat and r.scored]
        if not rs:
            continue
        typed = [r for r in rs if r.robot_type_correct is not None]
        passed_rounds = [r.rounds for r in rs if r.passed and r.rounds]
        avg = f"{sum(passed_rounds) / len(passed_rounds):.2f}" if passed_rounds else "–"
        lines.append(f"| {cat} | {len(rs)} | {_pct(sum(r.passed for r in rs), len(rs))} | "
                     f"{_pct(sum(bool(r.robot_type_correct) for r in typed), len(typed))} | "
                     f"{_pct(sum(bool(r.expectations_met) for r in rs), len(rs))} | {avg} |")
    return lines


def render_markdown(records: list[Record], meta: RunMeta, run_name: str) -> str:
    scored = [r for r in records if r.scored]
    main = [r for r in scored if r.split == "main"]
    hold = [r for r in scored if r.split == "holdout"]
    status = Counter(r.status for r in records)
    out = [f"# TaskBot benchmark: {meta.backend} / {meta.model}", ""]
    if meta.is_fake:
        out += ["> [!WARNING]", f"> # {FAKE_WARNING}", ""]
    out += [f"Run `{run_name}`, started {meta.started or 'unknown'}; provider {meta.provider}.",
            f"Prompts: {len(records)} recorded — {status.get(DONE, 0)} scored, "
            f"{status.get(NO_FAKE, 0)} with no fake response (not counted), "
            f"{status.get(ERROR, 0)} errors and {status.get(RATE_LIMITED, 0)} rate-limited "
            "(not counted; rerun the same command to retry them).", "",
            "A prompt passes when an in-scope task ends with a design passing every check and "
            "simulation test, or an out-of-scope task is rejected without a design.", ""]
    out += ["## Summary", ""]
    if meta.is_fake:
        out += ["No headline pass rate: the fake backend's answers are scripted.", ""]
    else:
        passed_rounds = [r.rounds for r in main if r.passed and r.rounds]
        out += [f"- **Overall pass rate (main prompts): {_pct(sum(r.passed for r in main), len(main))}**",
                f"- Holdout pass rate (reported separately): {_pct(sum(r.passed for r in hold), len(hold))}"
                + ("" if any(r.split == "holdout" for r in records) else " — not run (use --split all)"),
                "- Average rounds to pass: "
                + (f"{sum(passed_rounds) / len(passed_rounds):.2f}" if passed_rounds else "–"), ""]
    oos = [r for r in scored if r.should_reject]
    wrong = [r for r in scored if not r.should_reject and r.rejected]
    out += [f"- Out-of-scope rejection accuracy: {_pct(sum(bool(r.rejection_correct) for r in oos), len(oos))} "
            f"rejected; {len(wrong)} in-scope prompt(s) wrongly rejected"
            + (f" ({', '.join(r.id for r in wrong)})" if wrong else ""),
            f"- Valid spec: {_pct(sum(bool(r.valid_spec) for r in scored if not r.should_reject and not r.rejected), sum(1 for r in scored if not r.should_reject and not r.rejected))} "
            "of in-scope prompts that weren't rejected",
            f"- Tokens: {sum(r.tokens_in for r in records)} in + {sum(r.tokens_out for r in records)} out; "
            f"JSON validation retries: {sum(r.json_retries for r in records)}; rate-limit waits: "
            f"{sum(r.rate_limit_waits for r in records)}; total time {sum(r.time_s for r in records):.0f} s", ""]
    out += ["## Pass rate per category (main prompts)", "", *_rate_table(main), ""]
    out += ["## Holdout (reported separately)", ""]
    if hold:
        out += [*_rate_table(hold), ""]
    elif any(r.split == "holdout" for r in records):
        out += ["No holdout prompt was scored (see the table below).", ""]
    else:
        out += ["Not run in this benchmark (use --split all).", ""]

    any_round, final = Counter(), Counter()
    for r in scored:
        for k in {(f["check"], f["cause"] or "none") for f in r.failed_any_round}:
            any_round[k] += 1
        for k in {(f["check"], f["cause"] or "none") for f in r.unfixed}:
            final[k] += 1
    out += ["## Most common failing checks", ""]
    if any_round or final:
        out += ["| Check | Cause | Prompts failing it in any round | Prompts still failing at the end |",
                "|---|---|---|---|"]
        for (check, cause), n in sorted((any_round | final).items(), key=lambda kv: (-kv[1], kv[0])):
            out.append(f"| {check} | {cause} | {any_round.get((check, cause), 0)} | {final.get((check, cause), 0)} |")
    else:
        out.append("No check failed.")
    out += ["", "## Unfixed failures", "",
            "Failures still present after the last round, grouped by cause, with the exact messages "
            "the model was given. Causes that stay here are messages the model doesn't act on.", ""]
    groups = defaultdict(list)
    for r in scored:
        for u in r.unfixed:
            groups[u["cause"]].append((r.id, u))
    if not groups:
        out.append("None.")
    for cause in sorted(groups, key=lambda c: (-len(groups[c]), c)):
        out += [f"### {cause} ({len(groups[cause])})", ""]
        for pid, u in groups[cause]:
            msg = " ".join(str(u["message"]).split())
            out.append(f"- **{pid}** `{u['check']}`: {msg}")
        out.append("")
    out += ["## Every prompt", "",
            "| Id | Split | Category | Result | Robot type | Expectations | Rounds | Tokens | JSON retries | Time (s) | Report |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        exp = "" if r.expectations_met is None else ("met" if r.expectations_met else
                                                       "missed: " + "; ".join(r.expectation_failures))
        rtype = (r.robot_type or "") + ("" if r.robot_type_correct is None else (" ✓" if r.robot_type_correct else " ✗"))
        link = f"[report]({run_name}/{r.report})" if r.report else ""
        out.append(f"| {r.id} | {r.split} | {r.category} | {_outcome(r)} | {rtype} | {exp.replace('|', '/')} | "
                   f"{r.rounds or ''} | {r.tokens_in + r.tokens_out} | {r.json_retries} | {r.time_s:g} | {link} |")
    errs = [r for r in records if r.status in (ERROR, RATE_LIMITED)]
    if errs:
        out += ["", "## Not scored: errors", ""] + [f"- **{r.id}** ({r.status}): {(r.error or '').splitlines()[0]}"
                                                   for r in errs]
    return "\n".join(out) + "\n"


def write_reports(records: list[Record], meta: RunMeta, results_dir: Path) -> tuple[Path, Path]:
    md, csv_path = results_dir / f"{meta.name}.md", results_dir / f"{meta.name}.csv"
    md.write_text(render_markdown(records, meta, meta.name), "utf-8")
    write_csv(records, csv_path, meta.name)
    return md, csv_path


# --------------------------------------------------------------------------- compare


class CompareError(ValueError):
    pass


def compare(old_csv: Path, new_csv: Path, allow_different_models: bool = False) -> str:
    def read(path):
        with Path(path).open(newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    old, new = read(old_csv), read(new_csv)
    who = [sorted({(r["backend"], r["model"]) for r in rows}) for rows in (old, new)]
    if who[0] != who[1] and not allow_different_models:
        raise CompareError(f"the runs are from different models ({who[0]} vs {who[1]}); "
                           "pass --allow-different-models to compare them anyway")
    o = {r["id"]: r for r in old if r["status"] == DONE}
    n = {r["id"]: r for r in new if r["status"] == DONE}
    both = [k for k in n if k in o]
    better = [k for k in both if o[k]["passed"] == "n" and n[k]["passed"] == "y"]
    worse = [k for k in both if o[k]["passed"] == "y" and n[k]["passed"] == "n"]
    rounds = [k for k in both if o[k]["passed"] == n[k]["passed"] == "y" and o[k]["rounds"] != n[k]["rounds"]]
    lines = [f"# Compare {Path(old_csv).name} → {Path(new_csv).name}", "",
             f"Old: {', '.join(' / '.join(w) for w in who[0])}; new: {', '.join(' / '.join(w) for w in who[1])}",
             f"Scored in both: {len(both)}. Pass: {sum(o[k]['passed'] == 'y' for k in both)} → "
             f"{sum(n[k]['passed'] == 'y' for k in both)}.", ""]

    def section(title, keys, fmt):
        lines.extend([f"## {title} ({len(keys)})", ""] + ([fmt(k) for k in keys] or ["None."]) + [""])

    section("Better (fail → pass)", better, lambda k: f"- {k} ({n[k]['category']}): was {o[k]['final_failures'] or o[k]['error'] or 'fail'}")
    section("Worse (pass → fail)", worse, lambda k: f"- {k} ({n[k]['category']}): now {n[k]['final_failures'] or n[k]['error'] or 'fail'}")
    section("Still passing, different rounds", rounds, lambda k: f"- {k}: {o[k]['rounds']} → {n[k]['rounds']} rounds")
    only = sorted(set(o) ^ set(n))
    section("Scored in only one run", only, lambda k: f"- {k}: {'old' if k in o else 'new'} only")
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.]+", "-", text).strip("-") or "model"


def make_client_factory(backend: str, model: str | None, rate_limit_retries: int,
                        backoff_s: float) -> Callable[[], LLMClient]:
    from taskbot.fake import fake_client
    from taskbot.llm import make_client
    from taskbot.schema import load_motor_catalog, supported_motors

    def factory() -> LLMClient:
        client = (fake_client(supported_motors(load_motor_catalog()), strict=True) if backend == "fake"
                  else make_client(backend, model))
        client.rate_limit_retries, client.backoff_s = rate_limit_retries, backoff_s
        return client

    return factory


def main(argv: list[str] | None = None, bench_dir: Path = Path("bench")) -> int:
    from taskbot.cli import BACKENDS

    ap = argparse.ArgumentParser(prog="bench/run.py", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", choices=BACKENDS, default="fake")
    ap.add_argument("--model", help="model name (default: TASKBOT_MODEL or the backend's default)")
    ap.add_argument("--split", choices=("main", "holdout", "all"), default="main",
                    help="main prompts (default), the holdout set, or both (scored separately)")
    ap.add_argument("--category", action="append", choices=CATEGORIES, help="repeatable")
    ap.add_argument("--limit", type=int, help="only the first N selected prompts")
    ap.add_argument("--concurrency", type=int, default=2, help="prompts run at once (default 2)")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--retries", type=int, default=4, help="waits after a rate-limit error, per call")
    ap.add_argument("--backoff", type=float, default=5.0, help="first rate-limit wait in s; doubles")
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--record", action="store_true", help="render sim GIFs (slow)")
    ap.add_argument("--name", help="run name (default <date>_<backend>_<model>); reuse it to resume")
    ap.add_argument("--prompts-dir", type=Path, default=bench_dir)
    ap.add_argument("--results-dir", type=Path, default=bench_dir / "results")
    ap.add_argument("--compare", nargs=2, metavar=("OLD_CSV", "NEW_CSV"))
    ap.add_argument("--allow-different-models", action="store_true")
    args = ap.parse_args(argv)

    if args.compare:
        try:
            print(compare(*args.compare, allow_different_models=args.allow_different_models))
        except CompareError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        return 0
    try:
        prompts = select(load_suite(args.prompts_dir, args.split), args.category, args.limit)
        factory = make_client_factory(args.backend, args.model, args.retries, args.backoff)
        probe = factory()  # fails fast on missing configuration
    except (PromptFileError, ValueError, LLMError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    name = args.name or f"{dt.date.today().isoformat()}_{args.backend}_{_slug(probe.usage.model)}"
    meta = RunMeta(args.backend, probe.usage.model, probe.usage.provider, name)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Benchmark {name}: {len(prompts)} prompts, concurrency {args.concurrency}", file=sys.stderr)
    if meta.is_fake:
        print(f"WARNING: {FAKE_WARNING}", file=sys.stderr)
    try:
        records = run_bench(prompts, args.results_dir / name, factory, meta, concurrency=args.concurrency,
                            max_rounds=args.max_rounds, simulate=not args.no_sim, record=args.record,
                            progress=lambda line: print(line, file=sys.stderr))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    md, csv_path = write_reports(records, meta, args.results_dir)
    pending = [r.id for r in records if r.status not in FINAL]
    print(f"Wrote {md} and {csv_path}")
    if pending:
        print(f"{len(pending)} prompt(s) not finished ({', '.join(pending)}); rerun with --name {name} to resume.")
    return 0
