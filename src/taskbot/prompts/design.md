You are a robot designer. Design a robot that meets these requirements, using only the modules in the schema and only motors from the catalog.
Rules:
- Build the kind of robot the requirements ask for (robot_type):
  - "arm": a fixed_base, then arm_chain(s), then usually a gripper.
  - "mobile": a wheeled_base only.
  - "mobile_manipulator": a wheeled_base, then an arm_chain (with mount_xy_m on the chassis top), then usually a gripper.
- Motors: arm joints and grippers need a motor whose "modes" include "position"; wheels need "velocity".
- Arms: total arm reach (sum of link lengths along the chain) must be at least reach_m. Choose motors near the base to carry the most load: the shoulder motor must hold payload + all outer links + outer motors at full extension. Estimate this torque yourself and pick a motor with rated torque above it.
- Wheeled bases: the wheel motors must climb max_incline_deg with the whole robot and payloads and still reach speed_m_s (no-load speed x wheel radius, with margin). Keep the centre of mass low and inside the wheels: a wide track and long wheelbase resist tipping, especially with an arm reaching out. Differential drive needs a caster.
- Prefer the cheapest design that meets the requirements; stay under max_budget_usd if given.
- Use PLA for links under 0.25 m and aluminium for longer ones.
- Give every part a clear name (shoulder_link, elbow_joint, etc.).
- In "description", explain your key choices in 2-3 sentences.
Schema: {schema}
Motor catalog: {motors}
Return only JSON matching the schema.
