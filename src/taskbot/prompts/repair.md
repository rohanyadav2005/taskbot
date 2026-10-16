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
  - compliance: the error happened without any motor saturating, so the joints' position loops are too soft for the required precision; the message ranks the fixes for this design: a stiffer motor (higher Kp) on the joint that sags most, setting "control": "pid", shorter or lighter outer links, or bringing the load closer to the base.
  - unreachable: the target is outside what the arm can reach; lengthen links or add a joint.
  - unstable: the robot would tip over; widen the track or lengthen the wheelbase, lower or weigh down the base, move the arm mount towards the middle of the wheels, or shorten the reach.
  - slip: the driven wheels would lose grip; put more weight over them, lower the centre of mass, or drive all four wheels.
  - wrong_type: the design is a different kind of robot from the one the requirements ask for; rebuild it as that kind.
- In "description", add one line explaining what you changed and why.
Current spec: {spec}
Requirements: {requirements}
Return the complete revised spec as JSON.
