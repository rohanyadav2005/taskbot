"""The demo gallery: real recorded runs (qwen2.5:7b-instruct via Ollama) replayed with the replay
backend, so their results are real but need no LLM."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNS = ROOT / "tests" / "fixtures" / "real_runs"
MEDIA = ROOT / "docs" / "media"


@dataclass(frozen=True)
class Demo:
    key: str
    title: str
    prompt: str
    transcript: Path
    expected: str          # the outcome the recorded run reaches
    gif: Path | None = None

    @property
    def available(self) -> bool:
        return self.transcript.is_file()


DEMOS: list[Demo] = [
    Demo("windowsill", "Windowsill watering arm", "water the three plants on my windowsill",
         RUNS / "smoke_2026-11-07" / "1_windowsill.jsonl", "PASSED", MEDIA / "windowsill_arm.gif"),
    Demo("m3_screws", "M3 screw picker (2 mm)",
         "a desk arm that picks M3 screws out of a tray and drops them into a jar next to it",
         RUNS / "smoke_2026-11-07" / "2_m3_screws.jsonl", "PASSED", MEDIA / "m3_screws_arm.gif"),
    Demo("box", "2 kg box carrier", "a robot that carries a 2 kg box across my office",
         RUNS / "smoke_2026-11-07" / "3_box.jsonl", "PASSED", MEDIA / "box_carrier.gif"),
    Demo("toys", "Toy picker (mobile manipulator)", "a robot that picks toys off the floor and drops them in a basket",
         RUNS / "smoke_2026-11-07" / "4_toys.jsonl", "PASSED", MEDIA / "toy_picker.gif"),
    Demo("socks", "Sock collector", "Pick up dirty socks from the bedroom floor and put them in the laundry hamper",
         RUNS / "bench_2026-11-24" / "mm_sock_collector.jsonl", "PASSED"),
    Demo("pan", "Cast iron pan (correctly INFEASIBLE)",
         "Lift my 1.5 kg cast iron pan from the counter onto the stove burner",
         RUNS / "bench_2026-11-20" / "hard_arm_cast_iron_pan.jsonl", "INFEASIBLE"),
    Demo("drone", "Coffee drone (correctly OUT OF SCOPE)", "a flying drone that delivers coffee to my room",
         RUNS / "smoke_2026-11-07" / "6_drone.jsonl", "OUT OF SCOPE"),
]


def by_key(key: str) -> Demo:
    return next(d for d in DEMOS if d.key == key)


EXAMPLE_PROMPTS = [
    "water the three plants on my windowsill",
    "a robot that carries a 2 kg box across my office",
    "a robot that picks toys off the floor and drops them in a basket",
    "Sort a pile of LEGO bricks on my desk into four trays by colour",
    "Fetch the TV remote from the coffee table and bring it to me on the sofa",
    "Lift my 1.5 kg cast iron pan from the counter onto the stove burner",
]
