# Benchmark 2026-11-20 (real transcripts)

The first full benchmark with a real model: `bench/run.py` with qwen2.5:7b-instruct via Ollama on a
laptop CPU (Windows), hybrid mode, main prompts only. Seven of the 40 transcripts, copied unchanged
from `full_hybrid/out/<id>/<task>/llm_transcript.jsonl`; replayed by tests/test_bench_2026_11_20.py
with `--backend replay --transcript <file>`.

| File | Outcome then | Replayed now (Task 12) | What changed |
|---|---|---|---|
| mobile_office_box | INFEASIBLE | PASSED | "outdoor" for an office read as flat (regression from Task 11's ground clearance) |
| easy_arm_chess | INFEASIBLE | PASSED | a desk task as an arm; a chess piece weighs ~0.03 kg, not 0.5 |
| easy_arm_lego_sort | INFEASIBLE | PASSED | desk task as an arm; 3 g brick; autofix gives base_yaw a stiffer motor |
| easy_arm_cat_feeder | INFEASIBLE | PASSED | heights measured from the counter the arm stands on |
| mm_shelf_restock | INFEASIBLE | PASSED | the bottom shelf is at most ~0.4 m up |
| mm_tennis_balls | INFEASIBLE | PASSED | a court is hard and flat; a tennis ball ~0.06 kg |
| hard_arm_cast_iron_pan | INFEASIBLE | INFEASIBLE (correct), with an "In short" line on what would work | |
