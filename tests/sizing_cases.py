"""Generated requirements for the sizer's property test: feasible by construction (they pass
the Requirements validators and the feasibility gate), seeded so every run is the same."""

import numpy as np

from taskbot.feasibility import check_feasibility
from taskbot.requirements import Requirements


def arm_cases(n: int, catalog, seed: int = 7) -> list[Requirements]:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        reach = rng.uniform(0.2, 0.6)
        x_min = rng.uniform(0.1, 0.5 * reach)
        x_max = rng.uniform(x_min + 0.05, 0.9 * reach)
        half_y = rng.uniform(0.03, 0.5 * reach)
        z_min = rng.uniform(0.0, 0.1)
        z_max = z_min + rng.uniform(0.1, 0.35)
        kind = rng.choice(["parallel", "suction"])
        d = dict(task_summary=f"generated arm task {len(out)}", robot_type="arm", reach_m=round(reach, 3),
                 payload_kg=round(rng.uniform(0.0, 0.5), 3),
                 workspace={"x_min": round(x_min, 3), "x_max": round(x_max, 3), "y_min": round(-half_y, 3),
                            "y_max": round(half_y, 3), "z_min": round(z_min, 3), "z_max": round(z_max, 3)},
                 dof_min=3, gripper_needed=True, gripper_kind=str(kind), position_tolerance_m=0.01,
                 tolerance_reason="generated", assumptions=[])
        try:
            req = Requirements(**d)
        except ValueError:
            continue
        if check_feasibility(req, catalog).ok:
            out.append(req)
    return out


def mobile_cases(n: int, catalog, seed: int = 11) -> list[Requirements]:
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        d = dict(task_summary=f"generated mobile task {len(out)}", robot_type="mobile", payload_kg=0.0,
                 gripper_needed=False, speed_m_s=round(rng.uniform(0.2, 0.8), 2),
                 max_incline_deg=round(rng.uniform(0.0, 8.0), 1),
                 terrain=str(rng.choice(["flat", "carpet", "outdoor"])),
                 base_payload_kg=round(rng.uniform(0.0, 6.0), 2), position_tolerance_m=0.05,
                 tolerance_reason="generated", assumptions=[])
        req = Requirements(**d)
        if check_feasibility(req, catalog).ok:
            out.append(req)
    return out
