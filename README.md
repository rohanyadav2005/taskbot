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
joint ranges, budget, mass); once those all pass, it is simulated in MuJoCo
(`reach_targets`, `carry_payload`, `hold_still`). Failures go back to the LLM for
repair, up to `--max-rounds` (default 5). `report.md` shows each round's results,
a diff of what changed, the final pass/fail and GIFs of the final round's
simulations (`sim/*.gif`). Exit code 0 = passed, 2 = never passed.
`--no-sim` skips simulation; `--no-record` skips the GIFs.

Recording needs an OpenGL backend. On a headless Linux machine install one and
pick it before running, e.g. `apt install libosmesa6` and `export MUJOCO_GL=osmesa`
(or `libegl1` and `MUJOCO_GL=egl` with a GPU). Without one, the GIFs are skipped
and the report says why.

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
| `src/taskbot/sim/` | Scene building, IK, PD + gravity-compensation control, GIF recording |
| `src/taskbot/sim_tests.py` | Simulation tests generated from the requirements |
| `src/taskbot/repair.py` | design → check → simulate → repair loop |
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
| `joint_speed` | Every planned move can finish within 5 s with no joint above 80% of its no-load speed |
| `budget` | Motor prices + rough material cost ≤ `max_budget_usd` |
| `mass` | Always passes; warns if a desk-scale robot (reach ≤ 1 m) is over 5 kg |

Reach uses 2000 random poses and then refines the nearest ones with IK, because random
samples alone are too sparse for a 1 cm tolerance. All sampling uses fixed seeds.

## Simulation tests

Generated from the requirements; run only after every static check passes.

| Test | What happens | Passes when |
|---|---|---|
| `reach_targets` | IK to the workspace centre and 8 corners; a minimum-jerk move to each | Every target: error < 1 cm by move end + 1 s and staying there, overshoot ≤ 2 cm |
| `carry_payload` | Same moves with a `payload_kg` cube welded to the tool | As above, and no joint saturates for > 0.2 s |
| `hold_still` | Hold the target pose with the highest torque load, with payload, for 5 s | Tool drift < 5 mm (reported in µm) |

**Motors.** Available torque follows a linear torque-speed line: `rated * max(0, 1 - |w| / no_load_speed)`
when driving the joint the way it is turning; braking is limited to rated torque. Never stall torque.

**Move durations** are planned per move: `max(2 s, time so no joint exceeds 80% of its no-load speed)`.
The `joint_speed` check (static, IK only) fails if any move would need more than 80% of a motor's
no-load speed to finish within 5 s, naming the joint, motor and speeds.

**Model error.** `carry_payload` and `hold_still` are judged with the controller believing the
payload is 20% lighter than it is (`payload_mass_error=0.2`). The same tests with the true mass are
shown in the report as "(ideal)" rows, for information. A `hold_still` drift of exactly 0 raises
`SimMeasurementError`: it would mean the test measured nothing.

**Control** is joint-space PID: gains from each joint's effective inertia (plus payload),
8 Hz bandwidth, critically damped, 0.3 s integral time with anti-windup; gravity compensation
from `qfrc_bias` plus the believed payload weight. Physics runs at 1 ms.
