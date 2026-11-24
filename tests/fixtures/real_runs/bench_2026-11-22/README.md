# Benchmark 2026-11-22 (real transcripts)

The second full benchmark (after the scene-word fixes): `bench/run.py` with qwen2.5:7b-instruct via Ollama on a
laptop CPU (Windows), hybrid mode, main prompts only (`full_hybrid_v2`). Five of the 40
transcripts, copied unchanged from `out/<id>/<task>/llm_transcript.jsonl`; replayed by
tests/test_bench_2026_11_22.py with `--backend replay --transcript <file>`.

| File | Outcome then | Replayed after the second round of fixes | What changed |
|---|---|---|---|
| easy_arm_coffee_button | internal error (SimMeasurementError: hold_still zero deviation), labelled "invalid spec" | PASSED | 0 kg payload: zero sag is real; internal errors are labelled as TaskBot bugs |
| mm_tennis_balls | INFEASIBLE (outdoor: 0.08 m wheels, slip) | PASSED | a sports court is hard and level (flat) |
| mm_cafe_cups | INFEASIBLE (3 Nm vs 2.83) | PASSED | "a plastic cup" is 0.015 kg, not a 0.3 kg mug; targets near table and counter tops |
| vague_kitchen | INFEASIBLE (2.96 Nm vs 2.83) | PASSED | "a light plastic cup" is 0.015 kg |
| vague_mornings | INFEASIBLE (a 0.3 kg mug at 0.52 m: 3.2 Nm) | INFEASIBLE, saying what TaskBot needs to know | vague task line |
