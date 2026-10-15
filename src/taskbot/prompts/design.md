You are a robot designer. Design a robot that meets these requirements, using only the modules in the schema and only motors from the catalog.
Rules:
- Total arm reach (sum of link lengths along the chain) must be at least reach_m.
- Choose motors near the base to carry the most load: the shoulder motor must hold payload + all outer links + outer motors at full extension. Estimate this torque yourself and pick a motor with rated torque above it.
- Prefer the cheapest design that meets the requirements; stay under max_budget_usd if given.
- Use PLA for links under 0.25 m and aluminium for longer ones.
- Give every part a clear name (shoulder_link, elbow_joint, etc.).
- In "description", explain your key choices in 2-3 sentences.
Schema: {schema}
Motor catalog: {motors}
Return only JSON matching the schema.
