# TaskBot

Text-to-robot design with an LLM in the loop: plain-English task → measurable
requirements → module JSON (Pydantic schema) → deterministic URDF →
engineering checks and MuJoCo tests → plain-language feedback to the LLM.

```bash
pip install -e ".[dev]"
pytest
python -m taskbot.view examples/arm_4dof.json      # opens the MuJoCo viewer
```

## Layout

| Path | What |
|---|---|
| `src/taskbot/schema.py` | `RobotSpec` and its modules (`fixed_base`, `arm_chain`, `gripper`) — the only thing the LLM writes |
| `src/taskbot/data/motors.json` | Motor catalog with a source for every value |
| `src/taskbot/build_urdf.py` | `RobotSpec` → URDF; all mass and inertia computed here |
| `src/taskbot/view.py` | Spec JSON → URDF → MuJoCo viewer |
| `src/taskbot/checks.py` | `CheckResult`, returned by every check |
| `examples/` | 2-, 4- and 6-DOF arms |

## Geometry conventions

* Modules are listed root-first: one `fixed_base`, one or more `arm_chain`s, then at most one `gripper`.
* Each link extends along its own +z from the joint that drives it; the next joint is at its tip.
* A joint's motor is mounted on the parent link at the joint.
* URDF effort/velocity limits are the motor's stall torque and no-load speed.
  Prismatic joints assume a 10 mm rack-and-pinion radius.
