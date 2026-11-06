"""Property test: 200 seeded runs of the hybrid pipeline with a sloppy model (the mistakes real
7B models make). 0 crashes; every run writes report.md and run.json; at least 90% reach a sized
design, and every other run is infeasible or out of scope with a stated reason."""

import json
from collections import Counter

from taskbot.pipeline import run_pipeline
from taskbot.schema import load_motor_catalog
from taskbot.sloppy import SloppyClient

RUNS = 200
SIZED = {"PASSED", "FAILED", "APPROXIMATE"}


def test_sloppy_model_never_crashes_the_pipeline(tmp_path, capsys):
    catalog = load_motor_catalog()
    status, crashes, unexplained = Counter(), [], []
    for seed in range(RUNS):
        client = SloppyClient(seed)
        try:
            result = run_pipeline(client, client.prompt, catalog, tmp_path / str(seed), simulate=False, record=False)
        except Exception as exc:  # noqa: BLE001 - the point is to count them
            crashes.append((seed, f"{type(exc).__name__}: {exc}"))
            continue
        assert (result.out_dir / "report.md").exists(), seed
        run = json.loads((result.out_dir / "run.json").read_text(encoding="utf-8"))
        status[run["status"]] += 1
        if run["status"] not in SIZED and not run["reason"]:
            unexplained.append(seed)
    sized = sum(status[s] for s in SIZED)
    with capsys.disabled():
        print(f"\nsloppy property test: {RUNS} runs, {len(crashes)} crashes; " + ", ".join(
            f"{k} {v}" for k, v in sorted(status.items())) + f"; {sized}/{RUNS} reached a sized design")
    assert crashes == []
    assert unexplained == []
    assert set(status) <= SIZED | {"INFEASIBLE", "OUT_OF_SCOPE"}
    assert sized >= 0.9 * RUNS
