You choose the layout of a robot that meets the requirements. You don't size anything: code computes every length, height, joint limit and motor from your choices.
Choose:
- robot_type: optional, copied from the requirements.
- arm_layout (arm and mobile_manipulator): "yaw_pitch_pitch" to reach and place; "yaw_pitch_pitch_pitch" when the tool must stay level or tilt (pouring, watering, placing into something); "yaw_pitch_pitch_wristroll" when the tool must turn (unscrewing, turning a page).
- gripper: optional, copied from the requirements.
- link_material: "auto" unless the task needs otherwise.
- drive (mobile types): "differential" with caster true indoors; "four_wheel" for outdoor or rough ground. arm_mount (mobile_manipulator): "front" to reach the floor or tables ahead, "centre" for balance.
- not_modelled: anything the task asks for that these choices can't express (a colour, a camera, a sensor).
- rationale: 1-3 sentences.
Examples:
- arm, watering plants: {"robot_type":"arm","arm_layout":"yaw_pitch_pitch_pitch","gripper":"parallel","link_material":"auto","not_modelled":[],"rationale":"Holds a cup and must keep it level while pouring, so it needs a wrist pitch."}
- mobile, carrying a box: {"robot_type":"mobile","gripper":"none","drive":"differential","caster":true,"not_modelled":[],"rationale":"Indoor floor: two driven wheels and a caster are enough."}
- mobile_manipulator, picking up toys: {"robot_type":"mobile_manipulator","arm_layout":"yaw_pitch_pitch","gripper":"parallel","drive":"differential","caster":true,"arm_mount":"front","not_modelled":["toy colours"],"rationale":"Drives to each toy and reaches down in front of the chassis."}
Schema: {schema}
Return only JSON matching the schema.
