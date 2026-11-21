"""What the task's own words say about the scene, for the requirements normaliser
(taskbot.normalize): typical masses of everyday objects, and words for fixed places, travel,
indoor places, outdoor places and slopes.

The 7B model in the first real benchmark (2026-11-20) put 0.25-0.5 kg in the gripper for a
chess piece, a LEGO brick, a tennis ball, a tomato or a packet of crisps ("includes the
gripper"; TaskBot adds the gripper itself), made desk and conveyor tasks into mobile
manipulators, and called an office and a small warehouse "outdoor".

OBJECT_KG: one object as the gripper holds it, typical household values. Design conventions
from memory, unverified; they only ever lower a payload that is far above them (MAX_RATIO),
and never when the task states a mass itself.
"""

from __future__ import annotations

import re

OBJECT_KG: dict[str, tuple[float, str]] = {
    # word pattern (lower case) -> (typical mass of one, what that is). Household, kitchen, office,
    # workshop, garden and toy items; typical values from memory, all unverified.
    # -- household
    r"(tv |television )?remotes?( controls?)?": (0.15, "a TV remote"),
    r"(mobile |smart)?phones?|smartphones?": (0.2, "a phone"),
    r"keys|keyrings?|bunch of keys": (0.05, "a bunch of keys"),
    r"wallets?|purses?": (0.1, "a wallet"),
    r"spectacles|reading glasses|sunglasses|eyeglasses": (0.03, "a pair of glasses"),
    r"books?|cookbooks?|novels?": (0.4, "a book"),
    r"magazines?|newspapers?": (0.2, "a magazine"),
    r"envelopes?|letters|postcards?": (0.02, "an envelope"),
    r"(postage )?stamps?": (0.001, "a postage stamp"),
    r"parcels?|packages?": (1.0, "a small parcel"),
    r"pens?|pencils?|markers?|crayons?": (0.02, "a pen"),
    r"socks?": (0.05, "a sock"),
    r"t-?shirts?|shirts?": (0.2, "a shirt"),
    r"towels?": (0.4, "a towel"),
    r"shoes?|slippers?": (0.35, "a shoe"),
    r"pillows?|cushions?": (0.5, "a pillow"),
    r"blankets?": (1.0, "a blanket"),
    r"(aa |aaa )?batter(y|ies)": (0.023, "an AA battery"),
    r"light ?bulbs?": (0.05, "a light bulb"),
    r"candles?": (0.2, "a candle"),
    r"tissues?|tissue box": (0.1, "a tissue box"),
    r"(playing )?cards?": (0.002, "a playing card"),
    r"dice": (0.005, "a die"),
    r"pills?|tablets? of": (0.001, "a pill"),
    r"medicine bottles?|pill bottles?": (0.1, "a pill bottle"),
    # -- kitchen
    r"(empty |coffee |tea )?(cups?|mugs?)": (0.3, "a ceramic mug or a cup of coffee"),
    r"(drinking )?glass(es)?( of (water|wine|juice|milk))?": (0.25, "a drinking glass"),
    r"plates?|dishes": (0.4, "a dinner plate"),
    r"bowls?": (0.3, "a bowl"),
    r"forks?|spoons?|knives|knife|cutlery|chopsticks": (0.04, "a piece of cutlery"),
    r"cans? of (soda|cola|beer|pop|drink)|soda cans?|drink cans?": (0.36, "a full drink can"),
    r"(can|tin)s? of (cat|dog) food|(cat|dog|pet) food|dry food|kibble": (0.1, "a can or a scoop of pet food"),
    r"(water )?bottles?( of water)?": (0.55, "a 500 ml bottle of water"),
    r"wine bottles?|bottles? of wine": (1.3, "a bottle of wine"),
    r"eggs?": (0.06, "an egg"),
    r"apples?|oranges?|pears?|peaches?": (0.2, "an apple"),
    r"bananas?": (0.12, "a banana"),
    r"strawberr(y|ies)|berr(y|ies)|grapes?|cherr(y|ies)": (0.015, "a berry"),
    r"tomato(es)?": (0.15, "a large tomato"),
    r"potato(es)?|onions?|carrots?": (0.2, "a potato"),
    r"(loaf|loaves) of bread|bread": (0.5, "a loaf of bread"),
    r"jars? of \w+|jars?": (0.4, "a jar of jam"),
    r"salt|pepper|spice jars?|shakers?": (0.1, "a salt shaker"),
    r"kettles?": (1.5, "a full kettle"),
    r"(frying|cast iron|non-stick|sauce) ?pans?|saucepans?|pots and pans": (1.2, "a frying pan"),
    r"(packets?|bags?) of crisps|crisps|chips packets?|snack bags?": (0.05, "a packet of crisps"),
    r"cereal box(es)?|boxes of cereal": (0.5, "a cereal box"),
    r"tea ?bags?|sugar cubes?|coffee (capsules?|pods?)": (0.005, "a tea bag"),
    r"sandwich(es)?|snacks?": (0.2, "a sandwich"),
    r"drinks?": (0.35, "a drink"),
    r"trays? of drinks": (2.0, "a tray with a few drinks"),
    # -- office
    r"staplers?": (0.2, "a stapler"),
    r"sheets? of paper|papers?|pages?|documents?": (0.005, "a sheet of paper"),
    r"notebooks?|notepads?|folders?|files": (0.25, "a notebook"),
    r"laptops?": (1.5, "a laptop"),
    r"(computer )?mouse|mice": (0.1, "a computer mouse"),
    r"keyboards?": (0.6, "a keyboard"),
    r"usb (sticks?|drives?)|memory sticks?|sd cards?": (0.01, "a USB stick"),
    r"scissors": (0.08, "a pair of scissors"),
    r"(sticky )?tape( rolls?)?|rolls? of tape": (0.1, "a roll of tape"),
    r"paper ?clips?|drawing pins?|thumbtacks?|push pins?": (0.001, "a paper clip"),
    r"sticky notes?|post-its?": (0.002, "a sticky note"),
    r"cameras?|webcams?": (0.5, "a small camera"),
    # -- workshop
    r"m\d screws?|screws?": (0.002, "a small screw"),
    r"nuts|washers?": (0.002, "a nut"),
    r"bolts?|nails?|rivets?": (0.01, "a bolt"),
    r"screwdrivers?": (0.1, "a screwdriver"),
    r"hammers?|mallets?": (0.6, "a hammer"),
    r"wrench(es)?|spanners?|pliers": (0.2, "a spanner"),
    r"(power |cordless )?drills?": (1.5, "a cordless drill"),
    r"soldering irons?": (0.05, "a soldering iron"),
    r"glue guns?": (0.2, "a glue gun"),
    r"(circuit|pc) boards?|pcbs?": (0.03, "a small circuit board"),
    r"(electronic )?components?|resistors?|capacitors?|leds?|chips": (0.001, "an electronic component"),
    r"(3d[- ]printed )?parts?|pieces? of plastic": (0.02, "a small printed part"),
    r"paint ?brush(es)?|brush(es)?": (0.05, "a paintbrush"),
    r"tape measures?|measuring tapes?": (0.2, "a tape measure"),
    r"tools?": (0.3, "a hand tool"),
    # -- garden
    r"seeds?": (0.001, "a seed"),
    r"seedlings?|cuttings?": (0.1, "a seedling in a cell"),
    r"(small )?(flower ?)?pots?|plant pots?": (0.5, "a small plant pot"),
    r"potted plants?|(house)?plants?": (1.0, "a potted plant"),
    r"trowels?|hand forks?": (0.2, "a trowel"),
    r"leaves|leaf": (0.002, "a leaf"),
    r"weeds?": (0.05, "a weed"),
    r"watering cans?": (2.0, "a filled watering can"),
    r"(garden )?hoses?|hose nozzles?": (0.2, "a hose nozzle"),
    r"pine ?cones?|acorns?|conkers?|stones?|pebbles?": (0.02, "a pebble"),
    # -- toys and games
    r"chess ?pieces?|chessmen|pieces on a chess ?board|chess": (0.03, "a weighted chess piece"),
    r"lego( bricks?)?|duplo|building (blocks?|bricks?)|bricks?": (0.003, "a LEGO brick"),
    r"tennis ?balls?|ping[- ]?pong balls?|golf balls?": (0.06, "a tennis ball"),
    r"(foot|basket|soccer|beach) ?balls?|balls?": (0.43, "a football"),
    r"toy cars?|matchbox cars?": (0.05, "a toy car"),
    r"dolls?|action figures?": (0.3, "a doll"),
    r"(teddy( bears?)?|soft toys?|plush(ies)?|stuffed (animals?|toys?))": (0.2, "a soft toy"),
    r"puzzle pieces?|jigsaw pieces?": (0.002, "a puzzle piece"),
    r"marbles?": (0.005, "a marble"),
    r"toys?": (0.2, "a child's toy"),
}
MAX_RATIO = 2.0  # a payload more than this many times the object's typical mass is lowered to it

# A stated mass: "2 kg", "1.5kg", "500 g", "1 litre" (a litre of water is a kilogram).
_STATED_MASS = re.compile(r"\b\d+(\.\d+)?\s*(kg|kilo\w*|g|grams?|l|litres?|liters?)\b")
_INDOOR = re.compile(r"\b(office|offices|house|home|room|rooms|kitchen|bedroom|living room|hall|hallway|corridor|"
                     r"shop|store|warehouse|aisles?|cafe|café|restaurant|factory|lab|laboratory|indoors?|"
                     r"apartment|flat|garage|desks?|sofa)\b")
_OUTDOOR = re.compile(r"\b(outdoors?|outside|gardens?|yards?|backyard|lawns?|paths?|pavements?|sidewalks?|"
                      r"driveways?|courts?|fields?|park|street|streets|road|roads|patio|allotment|"
                      r"vegetable beds?|flower beds?)\b")
# ---- robot type from the task's words (normalize: like the scope gate, code decides clear cases)
_PLACE = (r"(the |my |our |a |his |her |their )?(office|offices|room|rooms|living room|bedroom|house|home|flat|"
          r"apartment|floor|floors|hall|hallway|corridor|building|warehouse|aisles?|shop|store|cafe|café|"
          r"restaurant|kitchen|garden|yard|backyard|lawn|park|court|path|street|desks|tables)")
# Moving from place to place: a verb that means it, or across/around/... a place.
_TRAVEL_VERB = re.compile(r"\b(deliver\w*|fetch\w*|bring\w*|brings|drives?|driving|patrol\w*|roam\w*|wander\w*|"
                          r"carr(y|ies|ying)|transport\w*|collect\w*|follows? me|room to room)\b")
_TRAVEL_PLACE = re.compile(rf"\b(across|around|through|along|between|up and down|down|about) {_PLACE}\b"
                           r"|\b(off|from|on|across) (the |my |our )?(\w+ )?floor\b")
# Work at one fixed place.
_FIXED_WORDS = re.compile(r"\b(desk|desktop|table|tabletop|tray|trays|conveyor|workbench|bench|board|jig|stand|"
                          r"windowsill|sill|counter|countertop|stove|machine|next to it|beside it|next to (my|the))\b")
# Picking up, holding or handling an object: needs an arm.
_HANDLE = re.compile(r"\b(pick\w*|grab\w*|grasp\w*|fetch\w*|collect\w*|lift\w*|take|takes|taking|scoop\w*|"
                     r"pour\w*|sort\w*|hold|holds|holding|place|places|placing|put|puts|putting|stack\w*|"
                     r"restock\w*|clear|clears|clearing|tidy|press\w*|turn the|stick\w*|water|watering|"
                     r"solder\w*|move the|moves the|hand me|hands me|tidy)\b")


def robot_place(task: str) -> tuple[str | None, list[str]]:
    """("travel" | "fixed" | None, the words that decided it). Travel words decide when present: a
    task that delivers, fetches or crosses a room travels even if it names a table. Fixed-place
    words decide when no travel word is there. Neither: None (the LLM's choice stands)."""
    text = task.lower()
    travel = [m.group(0) for r in (_TRAVEL_VERB, _TRAVEL_PLACE) for m in r.finditer(text)]
    if travel:
        return "travel", travel
    fixed = [m.group(0) for m in _FIXED_WORDS.finditer(text)]
    return ("fixed", fixed) if fixed else (None, [])


def handles_objects(task: str) -> list[str]:
    """Words saying the robot must pick up, hold or handle something (so it needs an arm)."""
    return [m.group(0) for m in _HANDLE.finditer(task.lower())]


def robot_type_from_words(task: str) -> tuple[str | None, str]:
    """(robot type the task's words decide, why) or (None, "") when they don't."""
    place, words = robot_place(task)
    handle = handles_objects(task)
    quoted = ", ".join(f'"{w}"' for w in dict.fromkeys(words))
    if place == "fixed":
        return "arm", f"the task works at one fixed place ({quoted}) and names nowhere else to go"
    if place == "travel" and handle:
        hq = ", ".join(f'"{w}"' for w in dict.fromkeys(handle))
        return "mobile_manipulator", f"the task moves between places ({quoted}) and handles objects ({hq})"
    if place == "travel":
        return "mobile", f"the task moves between places ({quoted}) and picks nothing up"
    return None, ""


# Heights of everyday surfaces above the floor, for a mobile manipulator's targets when the reply
# gives none (design conventions, unverified).
SURFACE_Z_M = {r"coffee tables?": 0.45, r"sofas?|couch|beds?": 0.45, r"chairs?": 0.45,
               r"desks?|tables?|dining tables?": 0.75, r"counters?|countertops?|worktops?|kitchen counters?": 0.9,
               r"windowsills?|sills?": 0.9, r"floor": 0.0}


def surface_heights(task: str) -> list[tuple[str, float]]:
    """Surfaces the task names with their typical heights; the longest match wins at each place."""
    text = task.lower()
    hits = sorted((m.start(), -(m.end() - m.start()), m.end(), m.group(0), z) for pat, z in SURFACE_Z_M.items()
                  for m in re.finditer(rf"\b({pat})\b", text))
    out, taken = [], -1
    for start, _, end, words, z in hits:
        if start >= taken:
            out.append((words, z))
            taken = end
    return out


# An arm standing on a work surface. Its heights are measured from that surface (TaskBot puts the base
# at z = 0); the model measured them from the floor for a bin on the counter (cat feeder: z 0.6-0.8 m).
_ON_SURFACE = re.compile(r"\bon (my|the|a|our) (desk|table|kitchen counter|counter|countertop|workbench|bench|"
                         r"windowsill)\b")
# Targets this high above a desk or counter top are the floor-measured height of the top itself:
# desks and tables stand about 0.75 m, kitchen counters about 0.9 m (design conventions, unverified).
SURFACE_FLOOR_Z_M = 0.5
# "the bottom shelf": the lowest shelf of a shop or home shelving unit, at most about 0.4 m up
# (design convention, unverified).
BOTTOM_SHELF_Z_M = 0.4
_BOTTOM_SHELF = re.compile(r"\b(bottom|lowest|low) shel(f|ves)\b")
_HIGH_SHELF = re.compile(r"\b(top|high|higher|upper|highest) shel(f|ves)\b")
# Words that name a slope or a step, so the model's incline stands.
SLOPE = re.compile(r"\b(ramps?|slopes?|sloped|sloping|hills?|hilly|inclines?|inclined|steps?|thresholds?|"
                   r"kerbs?|curbs?|uphill|downhill|stairs?)\b")


def objects_in(task: str) -> list[tuple[str, float, str]]:
    """Every everyday object the task names, in order: (the words, typical mass of one, what). At each
    place the longest match wins ("reading glasses" over "glasses"), and matches don't overlap."""
    text = task.lower()
    hits = []
    for pattern, (kg, what) in OBJECT_KG.items():
        for m in re.finditer(rf"\b({pattern})\b", text):
            hits.append((m.start(), -(m.end() - m.start()), m.end(), m.group(0), kg, what))
    out, taken = [], -1
    for start, _, end, words, kg, what in sorted(hits):
        if start >= taken:
            out.append((words, kg, what))
            taken = end
    return out


def object_in(task: str) -> tuple[str, float, str] | None:
    """(the words, typical mass of one, what) for the first everyday object the task names."""
    found = objects_in(task)
    return found[0] if found else None


def object_named(name: str) -> tuple[str, float, str] | None:
    """The table entry for an object the model named ("a ceramic mug" -> mug), if any."""
    return object_in(name or "")


def states_a_mass(task: str) -> bool:
    return bool(_STATED_MASS.search(task.lower()))


def stated_kg(task: str) -> float | None:
    """The first mass the task states, in kg ("5 kg", "500 g", "1 litre" of water = 1 kg)."""
    m = _STATED_MASS.search(task.lower())
    if m is None:
        return None
    value = float(re.match(r"[\d.]+", m.group(0))[0])
    unit = m.group(2)
    return value / 1000 if unit in ("g", "gram", "grams") else value


# The robot carries something on its body ...
_CARRIES = re.compile(r"\b(carr(y|ies|ying)|transport\w*|deliver\w*|bring\w*|brings|fetch\w*|haul\w*|"
                      r"a full|on (its|the) (back|top|body|base))\b")


def carries(task: str) -> list[str]:
    """Words saying the robot carries a load on its body (so a base load stands)."""
    return [m.group(0) for m in _CARRIES.finditer(task.lower())]


_CARPET = re.compile(r"\b(carpet\w*|rugs?)\b")


def terrain_from_words(task: str) -> tuple[str | None, list[str]]:
    """("outdoor" | "carpet" | "flat" | None, the words): any outdoor place makes it outdoor (the
    roughest ground the robot drives on); otherwise an indoor place is flat, or carpet when the task
    says carpet; no place word: None."""
    text = task.lower()
    out = [m.group(0) for m in _OUTDOOR.finditer(text)]
    if out:
        return "outdoor", out
    inside = [m.group(0) for m in _INDOOR.finditer(text)]
    carpet = [m.group(0) for m in _CARPET.finditer(text)]
    if carpet:
        return "carpet", carpet
    return ("flat", inside) if inside else (None, [])



_BUDGET = re.compile(r"([$£€]\s*\d|\b\d+(\.\d+)?\s*(usd|dollars?|bucks|pounds|gbp|euros?|eur)\b|\bbudget\b|"
                     r"\b(costs?|costing|price[sd]?|spend|afford|cheap|cheaper|cheapest|under \d+)\b)")


def states_a_budget(task: str) -> bool:
    """The task names money: a price, a budget or a cost limit."""
    return bool(_BUDGET.search(task.lower()))


def outdoors(task: str) -> bool:
    return bool(_OUTDOOR.search(task.lower()))


def names_a_slope(task: str) -> bool:
    return bool(SLOPE.search(task.lower()))


def on_a_surface(task: str) -> bool:
    """The arm stands on a desk, table, counter, workbench or windowsill the task names."""
    return bool(_ON_SURFACE.search(task.lower()))


def bottom_shelf_only(task: str) -> bool:
    text = task.lower()
    return bool(_BOTTOM_SHELF.search(text)) and not _HIGH_SHELF.search(text)

