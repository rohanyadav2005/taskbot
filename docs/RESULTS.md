# TaskBot results

Final benchmark of the hybrid pipeline (`full_hybrid_v3`, 2026-11-24): qwen2.5:7b-instruct through
Ollama on a laptop CPU (Windows), 40 main prompts from `bench/prompts.yaml`, then the 10 holdout
prompts from `bench/holdout.yaml`, which were never used to change code or prompts. These are
real-model runs; the replay tests in `tests/fixtures/real_runs/` re-run some of the same transcripts
offline.

## Main set (40 prompts)

Pass 33/40 (82%), strict pass 27/40 (68%), correct outcome 37/40 (92%): 25 of 32 in-scope tasks
designed and passing every check and simulation, 4 correctly reported INFEASIBLE, 8/8 out-of-scope
tasks rejected without a design. 0 crashes, 0 JSON retries. 4,311 s for all 40 (mean 135 s,
median 124 s per in-scope prompt).

| Prompt | Category | Result | Expected | Correct | Strict | Time (s) |
|---|---|---|---|---|---|---|
| easy_arm_water_plants | easy_arm | PASS | pass | yes | yes | 254 |
| easy_arm_lego_sort | easy_arm | PASS | pass | yes | no | 122 |
| easy_arm_phone_timelapse | easy_arm | INFEASIBLE | pass | **no** | no | 184 |
| easy_arm_envelope_stamps | easy_arm | PASS | pass | yes | yes | 98 |
| easy_arm_cat_feeder | easy_arm | PASS | pass | yes | yes | 113 |
| easy_arm_chess | easy_arm | PASS | pass | yes | yes | 127 |
| easy_arm_coffee_button | easy_arm | PASS | pass | yes | no | 101 |
| easy_arm_page_turner | easy_arm | PASS | pass | yes | yes | 109 |
| hard_arm_m3_screws | hard_arm | PASS | pass | yes | yes | 121 |
| hard_arm_grandma_shelf | hard_arm | INFEASIBLE | infeasible | yes | no | 155 |
| hard_arm_cast_iron_pan | hard_arm | INFEASIBLE | infeasible | yes | no | 120 |
| hard_arm_pcb_soldering | hard_arm | PASS | pass | yes | no | 106 |
| hard_arm_tomato_sorter | hard_arm | PASS | pass | yes | no | 131 |
| hard_arm_jug_pour | hard_arm | INFEASIBLE | infeasible | yes | no | 144 |
| mobile_office_box | mobile | PASS | pass | yes | yes | 51 |
| mobile_mail_delivery | mobile | PASS | pass | yes | yes | 92 |
| mobile_laundry_basket | mobile | PASS | pass | yes | yes | 87 |
| mobile_warehouse_patrol | mobile | PASS | pass | yes | yes | 73 |
| mobile_garden_cart | mobile | INFEASIBLE | infeasible | yes | no | 193 |
| mobile_party_drinks | mobile | PASS | pass | yes | yes | 93 |
| mm_toy_picker | mobile_manipulator | PASS | pass | yes | yes | 133 |
| mm_sock_collector | mobile_manipulator | PASS | pass | yes | yes | 124 |
| mm_tennis_balls | mobile_manipulator | PASS | pass | yes | no | 123 |
| mm_cafe_cups | mobile_manipulator | PASS | pass | yes | yes | 146 |
| mm_shelf_restock | mobile_manipulator | PASS | pass | yes | yes | 181 |
| mm_remote_fetch | mobile_manipulator | PASS | pass | yes | no | 120 |
| vague_grandma | vague | PASS | pass | yes | yes | 122 |
| vague_kitchen | vague | INFEASIBLE | pass | **no** | no | 172 |
| vague_workshop | vague | PASS | pass | yes | yes | 238 |
| vague_mornings | vague | PASS | pass | yes | yes | 130 |
| vague_shop | vague | INFEASIBLE | pass | **no** | no | 199 |
| vague_desk | vague | PASS | pass | yes | yes | 148 |
| oos_coffee_drone | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |
| oos_aquarium_fish | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |
| oos_hull_inspection | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |
| oos_robot_dog | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |
| oos_humanoid_laundry | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |
| oos_stairs_tea | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |
| oos_crop_sprayer | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |
| oos_yard_turret | out_of_scope | rejected (out of scope) | out of scope | yes | yes | 0 |

**Strict** also requires the extracted requirements to meet the prompt's expectations (robot type,
reach, payload range, tolerance range, terrain). The six passes that missed one: lego_sort and
remote_fetch (tolerance looser than expected), pcb_soldering (5 mm, expected at most 1 mm),
coffee_button (0.375 kg payload, expected at most 0.2 kg), tomato_sorter (reach 0.48 m, expected at
least 0.5 m) and tennis_balls (terrain flat, expected outdoor: TaskBot reads a sports court as a
hard, level surface).

### The three wrong outcomes

- **easy_arm_phone_timelapse** (INFEASIBLE, expected pass): the model asked to pan the phone across
  a box 0.8 m wide around the arm (±0.4 m), so the phone (0.2 kg, 0.25 kg with the 25% margin) is
  held 0.6 m out: about 3.1 Nm at the shoulder against 2.83 Nm usable from the strongest catalog
  servo. The same prompt passed in v2, where the model asked for ±0.3 m. The code did not change the
  outcome: both transcripts replay to their own result on today's code
  (`tests/test_bench_2026_11_24.py`); the model samples at temperature 0.2.
- **vague_shop** (INFEASIBLE, expected pass): for "a helper robot for my small shop" the model chose
  moving "a small box" of 4 kg (3 kg on its retry): about 29 Nm needed. In v2 it chose a screw and
  passed. The report says what TaskBot would need to know (which object, how heavy, how high).
- **vague_kitchen** (INFEASIBLE, expected pass): "a robot for my kitchen" became fetching a 0.3 kg
  coffee mug 0.49 m out: 2.96 Nm against 2.83 Nm, just over. In v2's replay with "a light plastic
  cup" it passed.

All three are the model choosing a heavier or farther version of a vague or loosely worded task;
none is a crash or a check that misjudged a design.

## Holdout (10 prompts, unseen)

Pass 7/10 (70%), strict 6/10 (60%), 2/2 out-of-scope rejected, 0 crashes. The holdout prompts were
written with the main set and kept aside: no code or prompt was changed after seeing these results,
so this is the honest score on tasks TaskBot was not tuned on. The holdout file predates expected
outcomes, so every in-scope prompt there counts as "expected pass".

| Prompt | Task | Result |
|---|---|---|
| holdout_easy_arm_tea_bag | Dip a tea bag in my mug and take it out after three minutes | PASS in 1 round |
| holdout_easy_arm_seedlings | Water the seedlings in a 40 by 30 cm tray on my desk with a small spray bottle | PASS in 1 round |
| holdout_hard_arm_beads | Place 5 mm beads one by one onto the pegs of a bead pattern board | PASS in 1 round |
| holdout_hard_arm_dumbbells | Put my two 2 kg dumbbells back on the rack next to my bench after each workout | infeasible (stopped) |
| holdout_mobile_toolbox | Follow me around the garage carrying my 3 kg toolbox | PASS in 1 round |
| holdout_mm_shoes | Pick up the shoes left in the hallway and put them on the shoe rack by the door | infeasible (stopped) |
| holdout_mm_dog_toys | Gather my dog's chew toys from around the back lawn into her toy box | infeasible (stopped) |
| holdout_vague_office | a robot for my office | PASS in 1 round |
| holdout_oos_pool_walls | A robot that swims along the walls of my swimming pool scrubbing off algae | rejected (correct) |
| holdout_oos_biped_greeter | A robot that walks on two legs to the front door to greet my guests | rejected (correct) |

### The three holdout misses (left as they are)

- **holdout_hard_arm_dumbbells**: 2 kg dumbbells (2.5 kg with the margin) at 0.79 m need about 20 Nm
  at the shoulder; the strongest catalog servo gives 2.83 Nm. By the main set's standard this is a
  correct INFEASIBLE (like the cast iron pan), but the holdout labels say pass and were not changed
  after the fact.
- **holdout_mm_shoes**: the model's shoe (0.35 kg, 0.44 kg with the margin) held 0.57 m out needs
  2.9 Nm against 2.83 Nm: INFEASIBLE by a narrow margin. A lighter shoe estimate or a shorter reach
  would pass; nothing was tuned for it.
- **holdout_mm_dog_toys**: on outdoor lawn the base and arm grow; the sized arm needs 3.8 Nm at the
  shoulder at full extension with a 0.25 kg toy (2.83 Nm usable), and none of the 3 intents tried
  could be built.

## Progression

| Run | Mode | Pass | Strict | Correct outcome | Time per in-scope prompt |
|---|---|---|---|---|---|
| sample_llm (10 prompts) | llm: the model writes every number | 0/10 | 0/10 | – | ~21 min |
| v1 (2026-11-20) | hybrid | 22/40 | 17/40 | – | mean 154 s |
| v2 (2026-11-22) | hybrid | 29/40 | 24/40 | 33/40* | mean 155 s (one prompt slept 50,351 s; left out) |
| v3 (2026-11-24) | hybrid | 33/40 | 27/40 | 37/40 | mean 135 s |
| holdout v3 | hybrid | 7/10 | 6/10 | – | mean 180 s |

The llm mode was run on 10 prompts only (12,634 s in all). "Correct outcome" was added to the
benchmark after v1. \* v2's own report, before two expected outcomes were relabelled (cafe cups:
pass; garden cart: infeasible).
