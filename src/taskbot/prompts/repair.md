Your robot design failed engineering checks. Fix it.
Failed checks:
{failures}
Rules:
- Change only what's needed to fix these failures; don't redesign parts that work.
- For torque failures: use a stronger motor from the catalog, shorten or lighten outer links, or move to a lighter material. Reaching further means more torque, so check that your fix doesn't break reach.
- For reach failures: lengthen links or add a joint, then reconsider torque.
- Each failure may name a cause:
  - saturated: a joint needed more torque than its motor has; use a stronger motor, or shorter or lighter outer links.
  - speed_limited: the motor is too slow, so it runs out of torque at speed; use a faster motor or a different gear ratio.
  - compliance: the error happened without any motor saturating, so a stronger motor will not help; it is a control or stiffness problem.
  - unreachable: the target is outside what the arm can reach; lengthen links or add a joint.
- In "description", add one line explaining what you changed and why.
Current spec: {spec}
Requirements: {requirements}
Return the complete revised spec as JSON.
