# TaskBot: engineering details

The reference behind the [README](../README.md): how the sizer picks each number, the feasibility
gate, the checks and simulations, and their conventions.

## Robot types

| Type | When | Modules |
|---|---|---|
| `arm` | the work happens in one place | `fixed_base`, `arm_chain`s, optional `gripper` |
| `mobile` | travels and carries things on its body, never picks anything up | `wheeled_base` |
| `mobile_manipulator` | travels and picks, places or handles objects | `wheeled_base`, `arm_chain` on the chassis top, optional `gripper` |

Flying, swimming, legs, stairs, humanoids and anything meant to hurt people or animals are out of
scope (exit 3).

**Wheeled base**: `differential` (two driven wheels, optional caster) or `four_wheel` (skid
steering). The battery defaults to 0.25 kg per pair of driven wheels. Wheel motors must support
velocity mode; arm and gripper motors must support position mode.

**Terrain** (rubber-wheel estimates, unverified):

| Terrain | Friction μ | Rolling resistance c_rr |
|---|---|---|
| `flat` (hard floor) | 0.8 | 0.01 |
| `carpet` | 0.6 | 0.03 |
| `outdoor` (paths, short grass) | 0.5 | 0.05 |

## What the sizer computes

- **Arm**: a riser at mid-workspace height (0.04–0.35 m on a fixed base, a mast up to 0.9 m on a
  mobile manipulator), raised until `self_collision` passes. The upper arm and forearm reach the
  farthest target exactly. Links are hollow tubes with a 2 mm wall: PLA under 0.25 m, aluminium
  above. Joint limits come from IK over the targets, widened by 15°.
- **Motors**, chosen from the tip inward: the cheapest motor whose usable torque is at least 1.25 ×
  the holding torque, with enough torque at peak speed. At a tolerance of 3 mm or less it picks the
  stiffest motor that passes. If even that can't hold the tolerance, the run is INFEASIBLE and says
  the best precision any design reached.
- **Mobile bases**: the cheapest wheel motor, the smallest chassis and the smallest wheel that clear
  the terrain, checked by the driving sims. A mobile manipulator's base yaw is stiffened until the
  arm holds the tool while driving.

The sizer only accepts sim results within 70% of each test's limit, so it never picks a design that
barely passes. When no catalog part fits a layout, the loop tries another layout. The run is
INFEASIBLE only when every layout fails, and the report lists what blocked each one.

**Margins (each applied exactly once):**

| Margin | Value | Applied in |
|---|---|---|
| Payload | ×1.25 on the bare mass | `normalize.py` |
| Reach | ×1.10 on the farthest workspace corner → `reach_m` | `normalize.py` |
| Motor | usable torque ≥ 1.25 × holding torque (else the cheapest that passes) | `sizing.py` |
| Usable torque | 80% of rated torque | `checks.py` |

## Feasibility gate

Before any design, `feasibility.py` checks limits no design can beat:

- **Arms**: the payload plus the lightest gripper at the needed reach, with weightless links,
  against the strongest motor's usable torque.
- **Mobile**: slope, rolling resistance and 0.5 m/s² acceleration against the terrain's friction,
  and the load against the best wheel motor on four driven wheels.

If it fails, the requirements go back to the LLM once. If they still fail, the run stops as
INFEASIBLE (exit 4) with one plain line, for example: "holding 1.88 kg at 0.62 m needs about 12 Nm
at the shoulder; the strongest motor here gives 2.83 Nm."

## Fixing the model's requirements

The normaliser corrects common LLM mistakes from the task's own words (`scene.py`) and lists every
change in the report:

- **Reach** is recomputed from the workspace, and a workspace centred on the base is moved in front
  of it.
- **Robot type**: travel words (deliver, fetch, across a room) mean mobile, or a mobile manipulator
  if it also handles things; desk, table or tray words mean an arm.
- **Terrain**: garden, yard or path means outdoor; office, house or kitchen means flat, or carpet
  when the task says so.
- **Slope** is capped at 5° indoors and 8° outdoors unless the task names a ramp or hill.
- **Payload** comes from a table of about 100 everyday objects (typical masses, unverified) unless
  the task states a mass.
- **Heights** measured from the floor are moved to the table, and invented heights are capped.
- **Vague tasks**: the report says what TaskBot needs to know (which object, how heavy, how high).
- Targets beyond 1.5 m are cut short, and the result is APPROXIMATE at best, never PASSED.

## Repair loop

- The final design is the best round, not the last: fewest failing checks, then the smallest
  shortfall.
- The loop stops early on an `infeasible` check or when the model repeats an earlier design.
- `self_collision` failures come with verified fixes: a shorter riser, a narrower joint range or a
  smaller base.

## Static checks

| Check | Passes when |
|---|---|
| `static_torque` | Every joint's gravity torque (arm straight out, payload at the tool) ≤ 80% of rated |
| `reach` | The tool reaches `reach_m` within 1 cm |
| `workspace_corners` | All 8 workspace corners are reachable within 1 cm |
| `self_collision` | ≤ 5% of random poses have links colliding |
| `joint_limits` | Every revolute joint spans ≥ 90° and stays within ±2π |
| `joint_speed` | Every move can finish within 5 s at ≤ 80% of no-load speed |
| `torque_at_speed` | At peak speed, gravity torque is below what the motor gives at that speed |
| `budget` | Motor and material cost ≤ the budget |
| `mass` | Always passes; warns above 5 kg for a desk-scale robot |

Mobile robots add:

| Check | Passes when |
|---|---|
| `wheel_torque` | Each driven wheel can accelerate at 0.5 m/s² up the slope at the target speed |
| `top_speed` | 80% of no-load wheel speed exceeds the target speed |
| `tip_over` | The centre of mass stays inside the support area (shrunk 15%), arm in every direction, on the slope |
| `traction` | The climbing force is below friction × load on the driven wheels |
| `ground_clearance` | Chassis and wheel size clear the terrain's typical obstacles |

All random sampling uses fixed seeds.

## Simulation tests

They run only after every static check passes. `tol` is the task's position tolerance (5 mm when
unclear). Physics runs at 1 ms.

| Test | What happens | Passes when |
|---|---|---|
| `reach_targets` | Move to the workspace centre and 8 corners | Error < tol, path deviation < 3 tol, overshoot < 2 tol |
| `carry_payload` | Same moves with the payload welded to the tool | As above, and no joint saturates for > 0.2 s |
| `hold_still` | Hold the most loaded pose for 5 s | Final error < tol, peak deviation < 3 tol |
| `drive_straight` | Drive 1 m and stop | Heading error < 5°, distance error < 5% |
| `turn_in_place` | Turn 90° | Error < 5° |
| `incline` | Drive 0.5 m up the slope | Covers the distance, never slides back > 10 mm |
| `drive_with_arm_extended` | Drive 0.5 m with the arm out in 4 directions | Tilt < 20°, tool within tol |

For a mobile manipulator, the arm tests run with the base held fixed.

**Motors and control** are fixed per motor, never tuned per test, and listed in every report.
Torque falls with speed, following the datasheet. Dynamixels use their factory gains, and hobby
servos use one conservative setting. Back-EMF damping is included, and the gravity feedforward
assumes a payload 20% lighter than it is. `"control": "pid"` adds a fixed integral term on
Dynamixels.

**Failure causes** tell the LLM what went wrong: `saturated`, `speed_limited`, `compliance`,
`unreachable`, plus `unstable`, `slip` and `wrong_type` for mobile robots.

## Conventions

- Modules are listed root-first: one base, then arm chains, then at most one gripper.
- Each link extends along its own +z, with the next joint at its tip. A motor sits on the parent
  link at its joint.
- Link shapes are `box`, `cylinder` and `tube` (hollow, 2 mm wall by default).
- URDF limits are the motor's stall torque and no-load speed.

## Known technical limits

- **Simple sizing rules**: link radius is a rule of thumb, base and gripper sizes are fixed, and
  the motor search only picks from the catalog.
- **Three arm layouts** in hybrid mode. Adding one means a list in `ARM_LAYOUTS` and a name in
  `intent.ArmLayout`.
- **Wheel sizing uses simulation**, so sizing a mobile base takes seconds, not milliseconds.
- **Autofix only knows numeric fixes**; tipping and anything else goes to a revised layout.
- **No velocity feedforward**: joints lag the planned timing. The lag is reported but never fails a
  test.
- **PID wind-up**: the integral builds during moves and leaves a few mm of error. Future: integrate
  only while holding.
- **Steppers aren't modelled properly**, so they're marked unsupported in `motors.json`.
