Your robot design failed engineering checks. Fix it.
Failed checks:
{failures}
Rules:
- Change only what's needed to fix these failures; don't redesign parts that work.
- For torque failures: use a stronger motor from the catalog, shorten or lighten outer links, or move to a lighter material. Reaching further means more torque, so check that your fix doesn't break reach.
- For reach failures: lengthen links or add a joint, then reconsider torque.
- In "description", add one line explaining what you changed and why.
Current spec: {spec}
Requirements: {requirements}
Return the complete revised spec as JSON.
