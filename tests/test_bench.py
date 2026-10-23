"""The benchmark: prompt files, scoring, the runner (fake and mocked HTTP backends),
resuming, reports (with the fake warning) and compare mode."""

import csv
import json
from collections import Counter
from pathlib import Path

import pytest
from openai_mock import MockOpenAI

from taskbot import bench
from taskbot.bench import (FAKE_WARNING, BenchPrompt, CompareError, Record, RunMeta, compare,
                           load_prompts, load_suite, run_bench, score_expectations, select)
from taskbot.fake import _load
from taskbot.llm import FakeClient, LLMRateLimitError, OpenAICompatClient
from taskbot.requirements import Requirements

BENCH = Path(__file__).resolve().parent.parent / "bench"
FAKE_META = RunMeta("fake", "scripted", "fake", "test_run")


def by_id(*ids: str) -> list[BenchPrompt]:
    suite = {p.id: p for p in load_suite(BENCH, "all")}
    return [suite[i] for i in ids]


def fake_factory():
    return bench.make_client_factory("fake", None, rate_limit_retries=0, backoff_s=0.0)


# --------------------------------------------------------------------------- prompt files


def test_main_prompts_have_the_required_categories():
    prompts = load_suite(BENCH, "main")
    assert len(prompts) == 40
    assert Counter(p.category for p in prompts) == {
        "easy_arm": 8, "hard_arm": 6, "mobile": 6, "mobile_manipulator": 6, "vague": 6, "out_of_scope": 8}
    assert all(p.split == "main" for p in prompts)


def test_task_5_1_prompts_are_included():
    ids = {p.id for p in load_suite(BENCH, "main") if p.note == "from Task 5.1"}
    assert ids == {"easy_arm_water_plants", "hard_arm_m3_screws", "mobile_office_box",
                   "hard_arm_grandma_shelf", "oos_coffee_drone"}


def test_holdout_is_separate_and_covers_every_category():
    hold = load_suite(BENCH, "holdout")
    assert len(hold) == 10 and all(p.split == "holdout" for p in hold)
    assert set(p.category for p in hold) == set(bench.CATEGORIES)
    main_ids = {p.id for p in load_suite(BENCH, "main")}
    assert not main_ids & {p.id for p in hold}


def test_out_of_scope_prompts_must_be_rejected_and_cover_the_cases():
    oos = [p for p in load_suite(BENCH, "main") if p.category == "out_of_scope"]
    assert all(p.expect.should_reject for p in oos)
    text = " ".join(p.prompt.lower() for p in oos)
    for word in ("drone", "swims", "underwater", "four-legged", "humanoid", "stairs", "shoots"):
        assert word in text


def write_yaml(tmp_path, text, name="prompts.yaml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.mark.parametrize("entry, error", [
    ("- {id: a1, category: arms, prompt: hello there}", "category"),
    ("- {id: a1, category: easy_arm, prompt: hello there, colour: red}", "colour"),
    ("- {id: a1, category: out_of_scope, prompt: a drone please}", "should_reject"),
    ("- {id: a1, category: easy_arm, prompt: hello there, expect: {should_reject: true}}", "should_reject"),
    ("- {id: a1, category: easy_arm, prompt: hello there, expect: {payload_kg: [2, 1]}}", "min <= max"),
    ("- {id: A-1, category: easy_arm, prompt: hello there}", "id"),
    ("- {id: a1, category: easy_arm, prompt: hello there, split: holdout}", "split"),
    ("- {id: a1, category: out_of_scope, prompt: fly away, expect: {should_reject: true, robot_type: arm}}",
     "can't be combined"),
])
def test_invalid_prompt_entries_are_reported_by_id(tmp_path, entry, error):
    with pytest.raises(bench.PromptFileError, match=error) as exc:
        load_prompts(write_yaml(tmp_path, entry))
    assert "a1" in str(exc.value) or "A-1" in str(exc.value)


def test_duplicate_ids_and_bad_yaml_are_rejected(tmp_path):
    two = "- {id: a1, category: easy_arm, prompt: one}\n- {id: a1, category: easy_arm, prompt: two}"
    with pytest.raises(bench.PromptFileError, match="duplicate id 'a1'"):
        load_prompts(write_yaml(tmp_path, two))
    with pytest.raises(bench.PromptFileError, match="not valid YAML"):
        load_prompts(write_yaml(tmp_path, "- {id: [unclosed"))
    with pytest.raises(bench.PromptFileError, match="expected a list"):
        load_prompts(write_yaml(tmp_path, "id: a1"))


def test_same_prompt_in_main_and_holdout_is_rejected(tmp_path):
    write_yaml(tmp_path, "- {id: a1, category: easy_arm, prompt: Water my plants}")
    write_yaml(tmp_path, "- {id: h1, category: easy_arm, prompt: water   my plants}", "holdout.yaml")
    with pytest.raises(bench.PromptFileError, match="prompt used in both"):
        load_suite(tmp_path)


def test_select_by_category_and_limit():
    prompts = load_suite(BENCH, "main")
    assert [p.category for p in select(prompts, ["mobile"])] == ["mobile"] * 6
    assert len(select(prompts, None, 5)) == 5
    assert len(select(prompts, ["vague", "mobile"], 3)) == 3
    with pytest.raises(ValueError, match="unknown categories"):
        select(prompts, ["flying"])


# --------------------------------------------------------------------------- scoring


def test_score_expectations():
    (p,) = by_id("mobile_office_box")
    req = Requirements(**_load("mobile_requirements"))
    assert score_expectations(req, p.expect) == (True, [])
    heavy = Requirements(**(_load("mobile_requirements") | {"base_payload_kg": 9.0}))
    ok, misses = score_expectations(heavy, p.expect)
    assert ok and misses == ["base_payload_kg 9.0 kg, expected 2–3.5 kg"]
    (m3,) = by_id("hard_arm_m3_screws")
    ok, misses = score_expectations(Requirements(**_load("mobile_requirements")), m3.expect)
    assert ok is False and misses[0] == "robot_type mobile, expected arm"
    assert any(m.startswith("reach_m None") for m in misses)


# --------------------------------------------------------------------------- runner, fake backend


@pytest.fixture(scope="module")
def fake_run(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("bench") / "run"
    prompts = by_id("mobile_office_box", "mobile_laundry_basket", "oos_coffee_drone", "vague_grandma",
                    "oos_yard_turret")
    records = run_bench(prompts, run_dir, fake_factory(), RunMeta(**vars(FAKE_META)), concurrency=2)
    return run_dir, prompts, {r.id: r for r in records}


def test_fake_runner_scores_each_prompt(fake_run):
    run_dir, _, recs = fake_run
    box = recs["mobile_office_box"]
    assert (box.status, box.passed, box.valid_spec, box.robot_type_correct, box.expectations_met, box.rounds) == (
        "done", True, True, True, True, 2)
    assert {(f["check"], f["cause"]) for f in box.failed_any_round} == {
        ("wheel_torque", "speed_limited"), ("top_speed", "speed_limited")}
    assert box.unfixed == [] and (run_dir / box.report).exists()

    stuck = recs["mobile_laundry_basket"]
    # The fake sends the same design back every round: stopped on the second repeat (round 3).
    assert stuck.status == "done" and not stuck.passed and stuck.rounds == 3
    assert stuck.stop_reason.startswith("repeating designs: round 3 is the same as round 1")
    unfixed = {u["check"]: u for u in stuck.unfixed}
    assert unfixed["wheel_torque"]["cause"] == "saturated"
    assert "stronger wheel motor" in unfixed["wheel_torque"]["message"]


def test_prompts_without_a_fake_answer_are_not_failures(fake_run):
    _, _, recs = fake_run
    for pid in ("vague_grandma", "oos_yard_turret"):
        r = recs[pid]
        assert r.status == "no_fake_response" and not r.scored and r.rejection_correct is None


def test_out_of_scope_rejection_is_scored(fake_run):
    _, _, recs = fake_run
    drone = recs["oos_coffee_drone"]
    assert drone.rejected and drone.rejection_correct and drone.passed and drone.rounds == 0


def _client_for(prompt_text_to_requirements: dict, design: dict | None = None):
    def factory():
        def requirements(system, messages, schema):
            return prompt_text_to_requirements[messages[0]["content"]]
        return FakeClient({"Requirements": requirements, "RobotSpec": lambda *a: design})
    return factory


def test_out_of_scope_prompt_that_is_not_rejected_fails(tmp_path):
    (drone,) = by_id("oos_coffee_drone")
    mobile = _load("mobile_requirements")  # no OUT OF SCOPE assumption
    recs = run_bench([drone], tmp_path / "r", _client_for({drone.prompt: mobile}, _load("mobile_design_round2")),
                     RunMeta("x", "y", "z", "r"), simulate=False)
    r = recs[0]
    assert r.status == "done" and r.rejected is False and r.rejection_correct is False and not r.passed
    assert "should have been rejected as out of scope" in r.expectation_failures


def test_in_scope_prompt_that_is_rejected_fails(tmp_path):
    (box,) = by_id("mobile_office_box")
    oos = _load("out_of_scope_requirements") | {"task_summary": box.prompt}
    r = run_bench([box], tmp_path / "r", _client_for({box.prompt: oos}), RunMeta("x", "y", "z", "r"))[0]
    assert r.rejected and not r.passed and r.rejection_correct is None
    assert "rejected as out of scope, but TaskBot can build this" in r.expectation_failures


def test_invalid_design_json_is_scored_as_invalid_spec(tmp_path):
    (box,) = by_id("mobile_office_box")
    r = run_bench([box], tmp_path / "r", _client_for({box.prompt: _load("mobile_requirements")}, {"name": 3}),
                  RunMeta("x", "y", "z", "r"))[0]
    assert r.status == "done" and r.valid_spec is False and not r.passed and r.json_retries == 3
    assert r.unfixed[0]["cause"] == "invalid_json" and "no valid RobotSpec after 4 attempts" in r.error


# --------------------------------------------------------------------------- resuming


class Flaky(FakeClient):
    """Rate-limited on every call."""

    def _chat(self, system, messages, schema):
        raise LLMRateLimitError("429 slow down", retry_after_s=0)


def test_resume_reruns_only_unfinished_prompts(tmp_path):
    run_dir = tmp_path / "run"
    prompts = by_id("oos_coffee_drone", "mobile_office_box", "vague_kitchen")
    real = fake_factory()
    made = []

    def first_factory():  # the office-box prompt gets rate-limited, the others run normally
        made.append(1)
        return real() if len(made) != 2 else Flaky({})

    first = run_bench(prompts, run_dir, first_factory, RunMeta(**vars(FAKE_META)), concurrency=1)
    assert [r.status for r in first] == ["done", "rate_limited", "no_fake_response"]
    calls = []

    def second_factory():
        calls.append(1)
        return real()

    second = run_bench(prompts, run_dir, second_factory, RunMeta(**vars(FAKE_META)))
    assert len(calls) == 1  # only the rate-limited prompt ran again
    assert [r.status for r in second] == ["done", "done", "no_fake_response"] and second[1].passed
    lines = (run_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 4  # 3 + the re-run; the last line for an id wins


def test_resume_after_a_crash_mid_file(tmp_path):
    run_dir = tmp_path / "run"
    prompts = by_id("oos_coffee_drone", "vague_kitchen")
    run_bench(prompts[:1], run_dir, fake_factory(), RunMeta(**vars(FAKE_META)))
    calls = []
    real = fake_factory()
    run_bench(prompts, run_dir, lambda: calls.append(1) or real(), RunMeta(**vars(FAKE_META)))
    assert len(calls) == 1


def test_report_paths_use_forward_slashes(fake_run, tmp_path):
    run_dir, _, recs = fake_run
    assert recs["mobile_office_box"].report.startswith("out/mobile_office_box/")
    assert all("\\" not in (r.report or "") for r in recs.values())
    # State saved on Windows by an older version had backslashes; resuming fixes them.
    old = Record("p1", "mobile", "main", "p", "done", False,
                 report="out\\p1\\office_rover\\report.md")
    (tmp_path / "results.jsonl").write_text(json.dumps(vars(old)) + "\n", encoding="utf-8")
    assert bench._load_state(tmp_path / "results.jsonl")["p1"].report == "out/p1/office_rover/report.md"


def test_resume_refuses_a_different_model(tmp_path):
    (p,) = by_id("oos_coffee_drone")
    run_bench([p], tmp_path / "run", fake_factory(), RunMeta(**vars(FAKE_META)))
    with pytest.raises(ValueError, match="holds a run of fake / scripted"):
        run_bench([p], tmp_path / "run", fake_factory(), RunMeta("openai_compat", "llama", "x", "run"))


# --------------------------------------------------------------------------- mocked HTTP backend


def test_runner_with_mocked_openai_compat_backend(tmp_path):
    from test_openai_compat import _scripted_pipeline

    (box,) = by_id("mobile_office_box")
    with MockOpenAI([_scripted_pipeline]) as m:
        def factory():
            return OpenAICompatClient(model="mock-model", base_url=m.url, api_key=None)
        meta = RunMeta("openai_compat", "mock-model", "openai_compat (mock)", "mock")
        (r,) = run_bench([box], tmp_path / "mock", factory, meta, simulate=False)
    assert r.status == "done" and r.passed and r.rounds == 1 and r.valid_spec
    assert (r.tokens_in, r.tokens_out, r.json_retries) == (200, 100, 0)
    assert r.model == "mock-model"


def test_runner_records_rate_limited_prompt_from_http_429(tmp_path):
    (box,) = by_id("mobile_office_box")
    with MockOpenAI([{"status": 429, "headers": {"Retry-After": "0"}, "body": {"error": "quota"}}]) as m:
        def factory():
            c = OpenAICompatClient(model="mock-model", base_url=m.url, api_key=None, rate_limit_retries=2)
            c._sleep = lambda s: None
            return c
        (r,) = run_bench([box], tmp_path / "mock", factory, RunMeta("openai_compat", "mock-model", "p", "mock"))
    assert r.status == "rate_limited" and r.rate_limit_waits == 2 and len(m.requests) == 3


# --------------------------------------------------------------------------- reports


def test_fake_report_has_the_warning_and_no_headline_rate(fake_run, tmp_path):
    _, _, recs = fake_run
    md, csv_path = bench.write_reports(list(recs.values()), RunMeta(**vars(FAKE_META)), tmp_path)
    text = md.read_text(encoding="utf-8")
    assert FAKE_WARNING in "\n".join(text.splitlines()[:5])
    assert "Overall pass rate" not in text and "No headline pass rate" in text
    assert "2 with no fake response (not counted)" in text
    assert "### saturated" in text and "mobile_laundry_basket** `wheel_torque`" in text
    assert "[report](test_run/out/mobile_office_box/" in text
    assert "| mobile | 2 | 1/2 (50%)" in text


def test_real_backend_report_has_headline_holdout_and_unfixed_messages(fake_run, tmp_path):
    _, _, recs = fake_run
    rows = [Record(**{**vars(r), "backend": "openai_compat", "model": "m"}) for r in recs.values()]
    rows.append(Record("holdout_mobile_toolbox", "mobile", "holdout", "p", "done", False, passed=True, rounds=3,
                       rejected=False, valid_spec=True))
    rows.append(Record("mm_cafe_cups", "mobile_manipulator", "main", "p", "error", False, error="HTTP 500 boom"))
    text = bench.render_markdown(rows, RunMeta("openai_compat", "m", "p", "real"), "real")
    assert FAKE_WARNING not in text
    assert "**Overall pass rate (main prompts): 2/3 (67%)**" in text
    assert "Holdout pass rate (reported separately): 1/1 (100%)" in text
    assert "Out-of-scope rejection accuracy: 1/1 (100%)" in text
    assert "## Unfixed failures" in text and "Even at rest it is short" in text
    assert "## Not scored: errors" in text and "HTTP 500 boom" in text
    assert "| wheel_torque | saturated | 1 | 1 |" in text


def test_csv_has_one_row_per_prompt(fake_run, tmp_path):
    _, _, recs = fake_run
    path = tmp_path / "x.csv"
    bench.write_csv(list(recs.values()), path, "test_run")
    rows = {r["id"]: r for r in csv.DictReader(path.open(newline="", encoding="utf-8"))}
    assert set(rows) == set(recs)
    stuck = rows["mobile_laundry_basket"]
    assert stuck["passed"] == "n" and stuck["rounds"] == "3" and "wheel_torque:saturated" in stuck["final_failures"]
    assert rows["oos_coffee_drone"]["rejection_correct"] == "y"
    assert rows["vague_grandma"]["status"] == "no_fake_response"


# --------------------------------------------------------------------------- compare


def _csv(tmp_path, name, model, passed: dict):
    rows = [Record(pid, "mobile", "main", "p", "done", False, passed=ok, rounds=2 if ok else 5,
                   backend="openai_compat", model=model) for pid, ok in passed.items()]
    path = tmp_path / f"{name}.csv"
    bench.write_csv(rows, path, name)
    return path


def test_compare_shows_better_and_worse(tmp_path):
    old = _csv(tmp_path, "old", "llama", {"a": False, "b": True, "c": True, "d": False})
    new = _csv(tmp_path, "new", "llama", {"a": True, "b": False, "c": True, "e": True})
    text = compare(old, new)
    assert "## Better (fail → pass) (1)\n\n- a" in text
    assert "## Worse (pass → fail) (1)\n\n- b" in text
    assert "## Scored in only one run (2)" in text and "- d: old only" in text and "- e: new only" in text


def test_compare_refuses_different_models_unless_allowed(tmp_path, capsys):
    old = _csv(tmp_path, "old", "llama", {"a": False})
    new = _csv(tmp_path, "new", "gemini", {"a": True})
    with pytest.raises(CompareError, match="different models"):
        compare(old, new)
    assert "Better (fail → pass) (1)" in compare(old, new, allow_different_models=True)
    assert bench.main(["--compare", str(old), str(new)]) == 2
    assert "--allow-different-models" in capsys.readouterr().err
    assert bench.main(["--compare", str(old), str(new), "--allow-different-models"]) == 0


# --------------------------------------------------------------------------- CLI


def test_cli_end_to_end_with_fake_backend(tmp_path, capsys):
    args = ["--backend", "fake", "--category", "out_of_scope", "--limit", "2",
            "--prompts-dir", str(BENCH), "--results-dir", str(tmp_path), "--name", "cli"]
    assert bench.main(args) == 0
    err = capsys.readouterr().err
    assert "WARNING: FAKE BACKEND" in err and "2 prompts" in err
    md = (tmp_path / "cli.md").read_text(encoding="utf-8")
    assert md.startswith("# TaskBot benchmark: fake / scripted") and FAKE_WARNING in md
    meta = json.loads((tmp_path / "cli" / "meta.json").read_text(encoding="utf-8"))
    assert meta["backend"] == "fake" and meta["model"] == "scripted"
    # Rerunning the same command resumes: nothing left to do.
    assert bench.main(args) == 0
    assert "Resuming: 2 of 2 prompts already done." in capsys.readouterr().err


def test_bench_run_script_exists():
    assert (BENCH / "run.py").read_text(encoding="utf-8").count("taskbot.bench import main") == 1


# --------------------------------------------------------------------------- strict pass


def test_strict_pass_needs_the_expectations_met(tmp_path):
    """The fake M3-screws arm passes its checks, but with the canned 5 mm tolerance and 0.25 kg
    payload it misses the prompt's expectations: a pass, not a strict pass."""
    prompts = by_id("hard_arm_m3_screws", "mobile_office_box", "oos_coffee_drone")
    recs = {r.id: r for r in run_bench(prompts, tmp_path / "run", fake_factory(), RunMeta(**vars(FAKE_META)),
                                       simulate=False)}
    m3 = recs["hard_arm_m3_screws"]
    assert m3.passed and not m3.expectations_met and not m3.strict_pass
    assert recs["mobile_office_box"].strict_pass and recs["oos_coffee_drone"].strict_pass
    rows = [Record(**{**vars(r), "backend": "openai_compat", "model": "m"}) for r in recs.values()]
    text = bench.render_markdown(rows, RunMeta("openai_compat", "m", "p", "real"), "real")
    assert "**Overall pass rate (main prompts): 3/3 (100%)**" in text
    assert "**Strict pass rate (main prompts): 2/3 (67%)**" in text
    assert "| hard_arm | 1 | 1/1 (100%) | 0/1 (0%) |" in text
    bench.write_csv(rows, tmp_path / "x.csv", "real")
    row = next(csv.DictReader((tmp_path / "x.csv").open(newline="", encoding="utf-8")))
    assert (row["id"], row["passed"], row["strict_pass"]) == ("hard_arm_m3_screws", "y", "n")


def test_older_state_without_strict_pass_is_derived(tmp_path):
    d = vars(Record("a", "mobile", "main", "p", "done", False, passed=True, expectations_met=True))
    del d["strict_pass"]
    (tmp_path / "results.jsonl").write_text(json.dumps(d) + "\n", encoding="utf-8")
    assert bench._load_state(tmp_path / "results.jsonl")["a"].strict_pass is True
