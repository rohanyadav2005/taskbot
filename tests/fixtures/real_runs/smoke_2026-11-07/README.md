# Smoke run 2026-11-07 (real transcripts)

Recorded by `scripts/smoke.ps1 -Backend ollama -Model qwen2.5:7b-instruct` (Ollama on a laptop
CPU, Windows) at commit f06e72a; copied unchanged from `smoke/2026-11-07_015916/transcripts/`.
Replayed by tests/test_smoke_replay_2.py with `--backend replay --transcript <file>`.

| File | Outcome then | Replayed now |
|---|---|---|
| 1_windowsill | PASSED, 192 s | PASSED |
| 2_m3_screws | PASSED, 85 s | PASSED |
| 3_box | PASSED, 51 s | PASSED |
| 4_toys | INFEASIBLE, 345 s (sizing 110 s) | INFEASIBLE, every intent tried: qwen's 15° ramp needs 0.1 m wheels that push the floor targets out of the arm's reach; at 10° it PASSES |
| 5_grandmother_shelf | INFEASIBLE, 172 s | INFEASIBLE |
| 6_drone | OUT OF SCOPE, 0.7 s | OUT OF SCOPE |
