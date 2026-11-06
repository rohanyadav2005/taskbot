"""Scope gate: a deterministic check of the user's own words, run before (and independent of)
the requirements LLM.

TaskBot designs fixed arms and wheeled robots. A prompt that names one of the capabilities
below stops the run as OUT OF SCOPE with the words that matched, before any LLM call:

    flying  fly, flying, drone, quadcopter, aerial, airborne
    water   swim, underwater, boat, submarine
    legs    walk, walking, legged, biped, quadruped, humanoid, stairs
    harm    weapon, gun, shoot, attack, turret

Simple inflections count ("drones", "flies", "shooting"). A word is ignored when a negation
("not", "no", "never", "without", "non", "instead of", "n't") comes up to NEGATION_WINDOW words
before it in the same clause: "a wheeled robot, not a drone" and "no flying" pass the gate.
The requirements LLM's own "OUT OF SCOPE:" assumption stays as a second check
(Requirements.out_of_scope, after taskbot.normalize).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# category -> (word as listed, regex for it and its inflections)
VOCABULARY: dict[str, tuple[tuple[str, str], ...]] = {
    "flying": (("fly", r"fl(?:y|ies|ying|ew|own)"), ("drone", r"drones?"), ("quadcopter", r"quadcopters?"),
               ("aerial", r"aerial"), ("airborne", r"airborne")),
    "water": (("swim", r"sw(?:im|ims|imming|am)"), ("underwater", r"underwater"), ("boat", r"boats?"),
              ("submarine", r"submarines?")),
    "legs": (("walk", r"walk(?:s|ing|ed)?"), ("legged", r"legged"), ("biped", r"bipeds?|bipedal"),
             ("quadruped", r"quadrupeds?|quadrupedal"), ("humanoid", r"humanoids?"), ("stairs", r"stairs?|staircases?")),
    "harm": (("weapon", r"weapons?"), ("gun", r"guns?"), ("shoot", r"shoot(?:s|ing)?"),
             ("attack", r"attack(?:s|ing|ed)?"), ("turret", r"turrets?")),
}
WHY = {
    "flying": "it needs a robot that flies",
    "water": "it needs a robot that swims or works on or under water",
    "legs": "it needs a robot with legs (walking or climbing stairs)",
    "harm": "TaskBot never designs a robot meant to hurt, threaten or attack people or animals",
}
NEGATIONS = {"not", "no", "never", "without", "non", "nor", "instead"}
NEGATION_WINDOW = 3
_CLAUSE = re.compile(r"[.,;:!?()]|\bbut\b")
_WORD = re.compile(r"[a-z]+")


@dataclass
class ScopeVerdict:
    matches: dict[str, list[str]] = field(default_factory=dict)   # category -> words as written
    ignored: list[str] = field(default_factory=list)              # negated matches, e.g. "not a drone"

    @property
    def out_of_scope(self) -> bool:
        return bool(self.matches)

    def reasons(self) -> list[str]:
        """One line per category, each starting with the OUT OF SCOPE prefix."""
        from taskbot.requirements import OUT_OF_SCOPE_PREFIX

        return [f"{OUT_OF_SCOPE_PREFIX} {WHY[c]}: the task says " + ", ".join(f'"{w}"' for w in words)
                + " (TaskBot's scope gate, from the task's own words; TaskBot designs fixed arms and wheeled robots)"
                for c, words in self.matches.items()]


def scope_gate(prompt: str) -> ScopeVerdict:
    text = prompt.lower().replace("’", "'").replace("n't", " not")
    verdict = ScopeVerdict()
    for clause in _CLAUSE.split(text):
        words = _WORD.findall(clause)
        for i, w in enumerate(words):
            for category, entries in VOCABULARY.items():
                if not any(re.fullmatch(pattern, w) for _, pattern in entries):
                    continue
                before = words[max(0, i - NEGATION_WINDOW):i]
                if NEGATIONS & set(before):
                    verdict.ignored.append(" ".join(before + [w]))
                elif w not in verdict.matches.get(category, []):
                    verdict.matches.setdefault(category, []).append(w)
    return verdict
