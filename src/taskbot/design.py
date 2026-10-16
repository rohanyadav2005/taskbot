"""Requirements -> RobotSpec via the LLM."""

from __future__ import annotations

import json

from taskbot.llm import LLMClient
from taskbot.prompts import load_prompt
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec, supported_motors


def design_system_prompt(catalog: dict[str, Motor]) -> str:
    schema = json.dumps(RobotSpec.model_json_schema(), separators=(",", ":"))
    # Sources are for humans; the model only needs the numbers.
    motors = json.dumps(
        [m.model_dump(exclude={"sources", "supported", "unsupported_reason"})
         for m in supported_motors(catalog).values()], separators=(",", ":")
    )
    # str.replace, not str.format: both payloads are full of braces.
    return load_prompt("design").replace("{schema}", schema).replace("{motors}", motors)


def design_user_message(req: Requirements) -> str:
    return "Requirements:\n" + req.model_dump_json(indent=2)


def design_robot(client: LLMClient, req: Requirements, catalog: dict[str, Motor]) -> RobotSpec:
    return client.complete_json(
        design_system_prompt(catalog),
        design_user_message(req),
        RobotSpec,
        context={"motor_ids": set(supported_motors(catalog))},
    )
