# Real-run transcripts (reconstructed)

Replayed with `--backend replay --transcript <file>` and by tests/test_replay_acceptance.py.
These four are reconstructed from the run summaries of real qwen2.5:7b-instruct runs (Ollama,
CPU) on "water the three plants on my windowsill"; only the requirements step is recorded, the
model's reply repeated for every attempt as it was in the runs. Real transcripts written by
`taskbot design` (out/<name>/llm_transcript.jsonl) can be dropped in next to them.

- windowsill_reach_crash: reach_m 0.3 with a workspace corner (0.3, -0.15, 0.275) 0.358 m from
  the shoulder: "no valid Requirements after 4 attempts", no report.
- windowsill_reach_08_z15: reach_m 0.8 with a workspace up to 1.5 m high.
- windowsill_double_margin: payload_kg 0.5 whose assumption says 0.4 kg including the margin.
- windowsill_centred_on_base: workspace x/y -0.15 to 0.15, z 0 to 0.3: a box around the base.
