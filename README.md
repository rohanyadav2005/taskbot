# TaskBot

Text-to-robot design with an LLM in the loop: plain-English task → measurable
requirements → module JSON (Pydantic schema) → deterministic URDF →
engineering checks and MuJoCo tests → plain-language feedback to the LLM.

```bash
pip install -e ".[dev]"
pytest
python -m taskbot.view examples/arm_4dof.json      # opens the MuJoCo viewer
```

## Design a robot from a prompt

```bash
pip install -e ".[anthropic]"
export ANTHROPIC_API_KEY=sk-ant-...
taskbot design "water the three plants on my windowsill"
# -> out/<name>/requirements.json, spec.json, robot.urdf

taskbot design "..." --backend ollama --model qwen2.5:7b-instruct   # local, needs `ollama serve`
taskbot design "..." --backend fake                                   # offline, canned output
```

`--backend` defaults to `$TASKBOT_BACKEND` (else `anthropic`); `--model` defaults to
`$TASKBOT_MODEL` (else the backend's default). If a reply fails validation, the errors
are sent back to the model and it retries, up to 3 times.

## Layout

| Path | What |
|---|---|
| `src/taskbot/schema.py` | `RobotSpec` and its modules (`fixed_base`, `arm_chain`, `gripper`) — the only thing the LLM writes |
| `src/taskbot/data/motors.json` | Motor catalog with a source for every value |
| `src/taskbot/build_urdf.py` | `RobotSpec` → URDF; all mass and inertia computed here |
| `src/taskbot/view.py` | Spec JSON → URDF → MuJoCo viewer |
| `src/taskbot/llm.py` | `LLMClient.complete_json` + Anthropic, Ollama and fake backends |
| `src/taskbot/requirements.py` | `Requirements` model and the prompt → requirements step |
| `src/taskbot/design.py` | Requirements → `RobotSpec` step |
| `src/taskbot/prompts/` | System prompts |
| `src/taskbot/cli.py` | `taskbot design "<prompt>"` |
| `src/taskbot/checks.py` | `CheckResult`, returned by every check |
| `examples/` | 2-, 4- and 6-DOF arms |

## Geometry conventions

* Modules are listed root-first: one `fixed_base`, one or more `arm_chain`s, then at most one `gripper`.
* Each link extends along its own +z from the joint that drives it; the next joint is at its tip.
* A joint's motor is mounted on the parent link at the joint.
* URDF effort/velocity limits are the motor's stall torque and no-load speed.
  Prismatic joints assume a 10 mm rack-and-pinion radius.
