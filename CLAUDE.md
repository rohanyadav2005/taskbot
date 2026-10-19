# TaskBot: notes for coding agents

Text-to-robot design with an LLM in the loop: an LLM extracts requirements and writes a JSON
spec; deterministic code builds the URDF, runs engineering checks and MuJoCo sims, and sends
failures back to the LLM until the design passes or runs out of rounds. See README.md.

## Project rules

- Python 3.11+, package `taskbot` under `src/`, tests with pytest under `tests/`.
- The LLM only writes JSON matching the Pydantic schema. All geometry, mass and inertia come
  from deterministic code.
- SI units everywhere.
- Every check returns `CheckResult{check, passed, value, limit, cause, message}`, with a message
  specific enough for an LLM to fix: name the part, give the numbers, say what to change.
- Controllers come from motor datasheets and fixed documented presets. Never tune gains,
  tolerances or test settings, or loosen a check, to make a design pass. When a model fails,
  fix the messages and prompts, not the checks.
- Values from memory rather than a source are marked unverified.
- Small tested steps; don't build beyond the current task.

## Working in this repo

- Tests: `pytest -q` (about 2.5 minutes; MuJoCo sims). Lint: `python -m pyflakes src tests bench`.
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
