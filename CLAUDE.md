# TaskBot: notes for coding agents

Text-to-robot design with an LLM in the loop: an LLM extracts requirements and chooses a
DesignIntent (layout, gripper, drive); deterministic code sizes every number (taskbot/sizing.py),
builds the URDF, runs engineering checks and MuJoCo sims, applies known numeric fixes
(taskbot/autofix.py), and asks the LLM for a different intent only when those run out. The older
"llm" design mode (the LLM writes the whole spec) is kept for comparison. See README.md.

## Project rules

- Python 3.11+, package `taskbot` under `src/`, tests with pytest under `tests/`.
- The LLM chooses the layout; code sizes every number. The LLM only writes JSON matching a
  Pydantic schema; in the default (hybrid) mode that is a DesignIntent with no numbers in it, and
  all lengths, heights, joint limits, motors, geometry, mass and inertia come from deterministic code.
- SI units everywhere.
- Every check returns `CheckResult{check, passed, value, limit, cause, message}`, with a message
  specific enough for an LLM to fix: name the part, give the numbers, say what to change.
- Controllers come from motor datasheets and fixed documented presets. Never tune gains,
  tolerances or test settings, or loosen a check, to make a design pass. When a model fails,
  fix the messages and prompts, not the checks.
- Values from memory rather than a source are marked unverified.
- Small tested steps; don't build beyond the current task.

## Working in this repo

- Tests: `pytest -q` (about 6 minutes: MuJoCo sims and a 200-run sloppy-model property test). Lint:
  `python -m pyflakes src tests bench`.
- Headless rendering needs `MUJOCO_GL=osmesa` or `egl`; tests skip rendering if neither works.
- Everything must also run on Windows: pass `encoding="utf-8"` to every `read_text`,
  `write_text` and text `open`, and write paths into reports and CSVs with `.as_posix()`.
- No API keys in code, logs, reports or commits. Offline runs use `--backend fake`; tests
  use the fake backend or the mocked HTTP server in `tests/openai_mock.py`, never a real
  provider. Never claim real-model results from fake or mocked runs.
- Benchmark: `bench/run.py` (see README "Running the benchmark"). Don't look at or tune on
  `bench/holdout.yaml` results while improving prompts.
- When a change alters an existing test, say which test and why.

## Git

- Work on `main`. Author commits as
  `rohanyadav2005 <rohanyadav2005@users.noreply.github.com>`
  (`git -c user.name=rohanyadav2005 -c user.email=rohanyadav2005@users.noreply.github.com commit`).
- No Co-Authored-By or other assistant attribution in commits or pull requests.
