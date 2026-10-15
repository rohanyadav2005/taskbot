"""Open a RobotSpec JSON in the MuJoCo viewer.

    python -m taskbot.view examples/arm_4dof.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco

from taskbot.build_urdf import build_model, build_urdf
from taskbot.schema import RobotSpec


def load_spec(path: str | Path) -> RobotSpec:
    return RobotSpec.model_validate_json(Path(path).read_text("utf-8"))


def load_model(path: str | Path) -> mujoco.MjModel:
    """Spec JSON -> validated RobotSpec -> URDF -> compiled MuJoCo model (with EE site)."""
    return build_model(load_spec(path))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec", type=Path, help="RobotSpec JSON file")
    parser.add_argument("--urdf-out", type=Path, help="also write the generated URDF here")
    args = parser.parse_args(argv)

    spec = load_spec(args.spec)
    urdf = build_urdf(spec)
    if args.urdf_out:
        args.urdf_out.write_text(urdf, "utf-8")
    model = build_model(spec)

    from mujoco import viewer  # imported late: needs a display

    viewer.launch(model)


if __name__ == "__main__":
    main()
