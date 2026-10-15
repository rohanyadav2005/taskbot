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
# -> out/<name>/report.md, requirements.json, spec.json, robot.urdf, rounds/round_N/

taskbot design "..." --backend ollama --model qwen2.5:7b-instruct   # local, needs `ollama serve`
taskbot design "..." --backend fake                                   # offline, canned output
```

The design is checked (static torque, reach, workspace corners, self-collision,
joint ranges, budget, mass). Failed checks go back to the LLM for repair, up to
`--max-rounds` (default 5). `report.md` shows each round's results, a diff of
what changed, and the final pass/fail. Exit code 0 = passed, 2 = never passed.

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
| `src/taskbot/repair.py` | design → check → repair loop |
| `src/taskbot/report.py` | Writes `out/<name>/` and `report.md` |
| `src/taskbot/cli.py` | `taskbot design "<prompt>"` |
| `src/taskbot/checks.py` | Engineering checks and `CheckResult` |
| `examples/` | 2-, 4- and 6-DOF arms |

## Geometry conventions

* Modules are listed root-first: one `fixed_base`, one or more `arm_chain`s, then at most one `gripper`.
* Each link extends along its own +z from the joint that drives it; the next joint is at its tip.
* A joint's motor is mounted on the parent link at the joint.
* URDF effort/velocity limits are the motor's stall torque and no-load speed.
  Prismatic joints assume a 10 mm rack-and-pinion radius.

## Check details

| Check | Passes when |
|---|---|
| `static_torque` | Each arm joint's gravity torque — chain straight and horizontal, payload at the tool point — is at most 80% of the motor's rated torque |
| `reach` | Tool reaches `reach_m` from the base axis at mid-workspace height, within 1 cm |
| `workspace_corners` | All 8 workspace corners are reachable within 1 cm |
| `self_collision` | ≤ 5% of random in-limit poses (tool inside the workspace) have non-adjacent links interpenetrating |
| `joint_limits` | Every revolute joint spans ≥ `min_joint_range_rad` (default 90°) and stays within ±2π |
| `budget` | Motor prices + rough material cost ≤ `max_budget_usd` |
| `mass` | Always passes; warns if a desk-scale robot (reach ≤ 1 m) is over 5 kg |

Reach uses 2000 random poses and then refines the nearest ones with IK, because random
samples alone are too sparse for a 1 cm tolerance. All sampling uses fixed seeds.
