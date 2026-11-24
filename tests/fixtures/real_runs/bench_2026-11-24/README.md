# Benchmark 2026-11-24 (real transcripts)

The final benchmark (`full_hybrid_v3`, after Task 13): qwen2.5:7b-instruct via Ollama on a laptop
CPU (Windows), hybrid mode, main prompts. Three of the 40 transcripts, copied unchanged. Two are replayed by
tests/test_bench_2026_11_24.py next to the same prompts' v2 transcripts (bench_2026-11-22);
mm_sock_collector (PASSED) is in the web app's demo gallery (taskbot/ui/gallery.py).

| File | v2 reply (today's code) | v3 reply (today's code, and Task 12's code) | Why |
|---|---|---|---|
| easy_arm_phone_timelapse | PASSED | INFEASIBLE: 0.25 kg at 0.6 m needs ~3.1 Nm (2.83 usable) | qwen asked for a pan region ±0.4 m across the desk instead of ±0.3 m |
| vague_shop | PASSED (a small screw) | INFEASIBLE: 3.75 kg at 0.77 m needs ~29 Nm | qwen chose "a small box" of 4 kg (3 kg on retry) |

Same system prompt (sha256 9aba553f74654617), same task text; the model samples at temperature 0.2,
so its requirements differ between runs. Neither outcome changed with Task 13's code.
