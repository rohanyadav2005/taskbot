# Smoke run 2026-11-12 (real transcripts)

Recorded by `scripts/smoke.ps1 -Backend ollama -Model qwen2.5:7b-instruct` (Ollama on a Windows
laptop CPU) at commit 0f69222; copied unchanged from `smoke/2026-11-12_104055/transcripts/`.
Replayed by tests/test_smoke_replay_3.py.

| File | Outcome then | Replayed now |
|---|---|---|
| 1_windowsill | PASSED, 246 s | PASSED |
| 2_m3_screws | PASSED, 103 s | PASSED |
| 3_box | PASSED, 73 s | PASSED |
| 4_toys | PASSED, 520 s (sizing 146 s), on qwen's 4th intent: a four-wheel base that only just turned in place | PASSED on qwen's first intent (no invented 15° ramp or 4 kg load) |
| 5_grandmother_shelf | INFEASIBLE, 173 s | INFEASIBLE |
| 6_drone | OUT OF SCOPE | OUT OF SCOPE |
