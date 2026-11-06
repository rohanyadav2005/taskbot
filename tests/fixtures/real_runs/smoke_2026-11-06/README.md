# Smoke run 2026-11-06 (real transcripts)

Recorded by `scripts/smoke.ps1 -Backend ollama -Model qwen2.5:7b-instruct` (Ollama on a laptop
CPU, Windows) at commit 1f728fa; copied unchanged from `smoke/2026-11-06_134522/transcripts/`.
Replayed by tests/test_smoke_replay.py with `--backend replay --transcript <file>`.

| File | Prompt | Outcome then | LLM time (of total) | Replayed now |
|---|---|---|---|---|
| 1_windowsill | water the three plants on my windowsill | INFEASIBLE | 293 s of 295 s | PASSED |
| 2_m3_screws | a desk arm that picks M3 screws out of a tray and drops them into a jar next to it | FAILED | 178 s of 184 s | PASSED (first intent) |
| 3_box | a robot that carries a 2 kg box across my office | PASSED | 119 s of 128 s | PASSED |
| 4_toys | a robot that picks toys off the floor and drops them in a basket | INFEASIBLE | 1009 s of 1009 s | PASSED (no intent was recorded: code's intents) |
| 5_grandmother_shelf | something to help my grandmother reach things on a high shelf | PASSED (rule-based fallback) | 2293 s of 2322 s | INFEASIBLE (shelf cut from 2.5 to 1.5 m, still too far) |
| 6_drone | a flying drone that delivers coffee to my room | FAILED | 211 s of 223 s | OUT OF SCOPE (scope gate, no LLM call) |
