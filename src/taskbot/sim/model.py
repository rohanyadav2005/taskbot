"""A simulation scene for one RobotSpec: robot + floor + light + optional welded payload."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from taskbot.build_urdf import (
    EE_SITE,
    LINEAR_DRIVE_RADIUS_M,
    build_mjspec,
    tool_link_name,
    tool_offset_m,
)
from taskbot.schema import (
    ArmChain,
    Gripper,
    Joint,
    Motor,
    RobotSpec,
    WheeledBase,
    gripper_names,
    load_motor_catalog,
    wheel_names,
)
from taskbot.sim.motor import MotorModel, motor_model

BASE_PAYLOAD_BODY = "base_payload"
PAYLOAD_BODY = "payload"
PAYLOAD_DENSITY_KG_M3 = 1000.0  # payload drawn as a water-density cube
FLOOR_FRICTION = 0.01           # the wheel / caster geom's own friction decides each contact
SIM_TIMESTEP_S = 0.001


@dataclass
class SimRobot:
    """The compiled scene plus index bookkeeping for the arm joints."""

    model: mujoco.MjModel
    joints: list[Joint]            # arm joints, root first
    qadr: np.ndarray               # qpos address per arm joint
    dadr: np.ndarray               # dof address per arm joint
    lower: np.ndarray              # joint limits; ±inf for continuous
    upper: np.ndarray
    motors: list[MotorModel]       # per arm joint (scaled to N, m/s for prismatic joints)
    site: int                      # end-effector site id
    tool_body: int
    payload_kg: float
    payload_body: int              # -1 when there is no payload
    aux_qadr: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # finger joints
    aux_dadr: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    aux_motors: list[MotorModel] = field(default_factory=list)
    # wheeled base (empty for a fixed base)
    wheel_names: list[str] = field(default_factory=list)
    wheel_qadr: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    wheel_dadr: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    wheel_act: np.ndarray = field(default_factory=lambda: np.zeros(0, int))
    wheel_side: np.ndarray = field(default_factory=lambda: np.zeros(0))  # +1 left, -1 right
    wheel_motors: list[MotorModel] = field(default_factory=list)
    chassis_body: int = -1
    free_qadr: int = -1        # qpos address of the chassis free joint, -1 if fixed

    @property
    def n(self) -> int:
        return len(self.joints)

    @property
    def torque_limit(self) -> np.ndarray:
        """Rated torque (N·m) or force (N) per arm joint."""
        return np.array([m.rated for m in self.motors])

    @property
    def no_load_speed(self) -> np.ndarray:
        """rad/s (or m/s for prismatic) per arm joint."""
        return np.array([m.no_load_speed for m in self.motors])

    def sample_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Limits with continuous joints mapped to [-π, π], for random sampling."""
        return np.where(np.isfinite(self.lower), self.lower, -np.pi), \
            np.where(np.isfinite(self.upper), self.upper, np.pi)

    def home(self) -> np.ndarray:
        """All joints at zero (arm straight up), clipped into the limits."""
        return np.clip(np.zeros(self.n), self.lower, self.upper)

    def set_pose(self, data: mujoco.MjData, q: np.ndarray) -> None:
        """Put the arm at q (at rest) and, if present, the payload in the gripper."""
        data.qpos[:] = self.model.qpos0
        data.qvel[:] = 0
        data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, data)
        if self.payload_body >= 0:
            adr = self.model.jnt_qposadr[self.model.body_jntadr[self.payload_body]]
            data.qpos[adr : adr + 3] = data.site_xpos[self.site]
            data.qpos[adr + 3 : adr + 7] = data.xquat[self.tool_body]
        mujoco.mj_forward(self.model, data)

    def ee(self, data: mujoco.MjData) -> np.ndarray:
        return data.site_xpos[self.site].copy()


def build_scene(
    spec: RobotSpec,
    catalog: dict[str, Motor] | None = None,
    payload_kg: float = 0.0,
    markers: list[np.ndarray] = (),
    floating: bool = False,
    terrain: str = "flat",
    base_payload_kg: float = 0.0,
    incline_deg: float = 0.0,
) -> SimRobot:
    """Robot with floor, light and target markers; a payload cube welded at the tool point
    when payload_kg > 0. Markers and the payload cube don't collide with anything.

    The floor is visual only unless `floating` (a mobile robot driving): then the chassis
    has a free joint and the floor is solid, with near-zero friction of its own so each
    contact takes the other geom's friction (MuJoCo uses the larger of the two): the
    terrain's for wheels, almost none for the caster. `base_payload_kg` is a box fixed on
    the chassis top. `incline_deg` tilts gravity so that +x is uphill.
    """
    catalog = catalog if catalog is not None else load_motor_catalog()
    mjspec = build_mjspec(spec, catalog, floating=floating, terrain=terrain)
    mjspec.option.timestep = SIM_TIMESTEP_S
    mjspec.visual.headlight.ambient = [0.45, 0.45, 0.45]
    mjspec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    world = mjspec.worldbody
    world.add_light(pos=[0.5, -0.5, 1.5], dir=[-0.3, 0.3, -1], diffuse=[0.5, 0.5, 0.5])
    if floating:
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[5, 5, 0.01], pos=[0, 0, 0],
                       rgba=[0.9, 0.9, 0.88, 1], friction=[FLOOR_FRICTION, 0, 0], condim=6)
    else:
        world.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[1, 1, 0.01], pos=[0, 0, -0.0005],
                       rgba=[0.9, 0.9, 0.88, 1], contype=0, conaffinity=0)
    if incline_deg:
        th = np.radians(incline_deg)
        mjspec.option.gravity = [-9.81 * np.sin(th), 0.0, -9.81 * np.cos(th)]
    base = spec.base
    if isinstance(base, WheeledBase) and base_payload_kg > 0:
        # Its own body with explicit inertia: the URDF chassis body has an explicit
        # <inertial>, so MuJoCo would ignore the mass of a geom added to it.
        side = (base_payload_kg / PAYLOAD_DENSITY_KG_M3) ** (1 / 3)
        box = mjspec.body(base.name).add_body(
            name=BASE_PAYLOAD_BODY, pos=[0.0, 0.0, base.chassis_top_m + side / 2],
            mass=base_payload_kg, inertia=[base_payload_kg * side ** 2 / 6] * 3, explicitinertial=True)
        box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[side / 2] * 3, rgba=[0.6, 0.4, 0.2, 1],
                     contype=0, conaffinity=0, mass=0)
    for i, p in enumerate(markers):
        world.add_site(name=f"target_{i}", pos=list(map(float, p)), size=[0.008, 0, 0],
                       rgba=[0.1, 0.7, 0.2, 0.8])

    tool = tool_link_name(spec) if spec.has_arm else None
    if payload_kg > 0 and tool is not None:
        # Place the cube at the tool point for the qpos0 pose so the weld's
        # reference relative pose is "cube centred on the tool point".
        probe = mjspec.compile()
        pd = mujoco.MjData(probe)
        mujoco.mj_kinematics(probe, pd)
        site_pos = pd.site_xpos[mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)]
        half = (payload_kg / PAYLOAD_DENSITY_KG_M3) ** (1 / 3) / 2
        body = world.add_body(name=PAYLOAD_BODY, pos=list(map(float, site_pos)))
        body.add_freejoint()
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[half] * 3, mass=payload_kg,
                      rgba=[0.95, 0.55, 0.1, 1], contype=0, conaffinity=0)
        weld = mjspec.add_equality(type=mujoco.mjtEq.mjEQ_WELD, name1=PAYLOAD_BODY, name2=tool,
                                   objtype=mujoco.mjtObj.mjOBJ_BODY)
        # Anchor at the tool point (in the tool link's frame), where the cube's centre
        # is; a zero relpose quaternion makes the compiler take the relative pose from
        # qpos0. MjSpec's default anchor is (0, 1, 0), a metre away.
        weld.data = [0.0, 0.0, tool_offset_m(spec)] + [0.0] * 7 + [1.0]
        # Stiffer than MuJoCo's default so the cube doesn't swing on the weld: the
        # time constant is the smallest stable one, two timesteps.
        weld.solref = [2 * mjspec.option.timestep, 1.0]
        weld.solimp = [0.95, 0.99, 0.001, 0.5, 2.0]
    segments = [s for m in spec.modules if isinstance(m, ArmChain) for s in m.segments]
    joints = [s.joint for s in segments]
    motors = []
    for j in joints:
        mm = motor_model(catalog[j.motor_id], spec.control)
        motors.append(mm.linear(LINEAR_DRIVE_RADIUS_M) if j.type == "prismatic" else mm)
    gripper = next((m for m in spec.modules if isinstance(m, Gripper)), None)
    gmotor = (motor_model(catalog[gripper.motor_id], spec.control).linear(LINEAR_DRIVE_RADIUS_M)
              if gripper is not None and gripper.motor_id else None)
    finger_joints = [n for n in gripper_names(gripper)[1][1:]] if gripper is not None else []
    # Back-EMF and D-term damping act as joint damping; MuJoCo integrates it implicitly.
    for j, mm in zip(joints, motors):
        mjspec.joint(j.name).damping = [mm.joint_damping, 0.0, 0.0]
    for name in finger_joints:
        mjspec.joint(name).damping = [gmotor.joint_damping, 0.0, 0.0]
    wheel_joints, wmotor = [], None
    if isinstance(base, WheeledBase):
        wmotor = motor_model(catalog[base.wheel_motor_id], mode="velocity")
        wheel_joints = [n for n in wheel_names(base)[1] if not n.endswith("_caster_mount")]
        for name in wheel_joints:
            mjspec.joint(name).damping = [wmotor.joint_damping, 0.0, 0.0]
    model = mjspec.compile()

    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j.name) for j in joints]
    aux = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in finger_joints]

    return SimRobot(
        model=model,
        joints=joints,
        qadr=np.array([model.jnt_qposadr[j] for j in jids], int),
        dadr=np.array([model.jnt_dofadr[j] for j in jids], int),
        lower=np.array([-np.inf if j.type == "continuous" else j.lower for j in joints]),
        upper=np.array([np.inf if j.type == "continuous" else j.upper for j in joints]),
        motors=motors,
        site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE),
        tool_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, tool) if tool else -1,
        payload_kg=payload_kg if tool is not None else 0.0,
        payload_body=(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, PAYLOAD_BODY)
                      if payload_kg > 0 and tool is not None else -1),
        aux_qadr=np.array([model.jnt_qposadr[j] for j in aux], int),
        aux_dadr=np.array([model.jnt_dofadr[j] for j in aux], int),
        aux_motors=[gmotor] * len(aux),
        wheel_names=wheel_joints,
        wheel_qadr=np.array([model.jnt_qposadr[model.joint(n).id] for n in wheel_joints], int),
        wheel_dadr=np.array([model.jnt_dofadr[model.joint(n).id] for n in wheel_joints], int),
        wheel_act=np.array([model.actuator(f"{n}_motor").id for n in wheel_joints], int),
        wheel_side=np.array([1.0 if "left" in n else -1.0 for n in wheel_joints]),
        wheel_motors=[wmotor] * len(wheel_joints),
        chassis_body=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base.name),
        free_qadr=(int(model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, base.name)]]) if floating else -1),
    )
