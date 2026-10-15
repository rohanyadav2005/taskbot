"""System prompts, loaded with `load_prompt(name)`."""

from importlib import resources


def load_prompt(name: str) -> str:
    return resources.files(__name__).joinpath(f"{name}.md").read_text("utf-8")
