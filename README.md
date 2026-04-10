# World's Hardest Game RL

A high-accuracy reinforcement learning + planning project for **The World's Hardest Game** with strict collision rules and full 30-level support.

## Full-Game Demo (All 30 Levels, Strict)

![All 30 levels strict run](rl/models/strict_timeout_levels1_30_flashmove_v3_all30.gif)

## Why This Repo Is Different

- Full 30-level Flash-derived environment
- Strict anti-exploit collision handling (edge-aware, no clip-through)
- Deterministic planning fallback for hard levels
- End-to-end all-30 strict solved rollout artifact

## Quick Start

### 1) Java clone (fast local play)

```bash
java -jar "World's Hardest Game.jar"
```

### 2) Level-1 RL baseline

```bash
python3 rl/train_agent.py
python3 rl/watch_agent.py --model rl/models/level1_qtable.npz
```

### 3) Full strict all-30 render

```bash
python3 rl/render_strict_timeout_sweep.py \
  --dataset-dir rl/data/flash_levels \
  --levels 1-30 \
  --enemy-hit-radius 6.0 \
  --planner-max-expand 1200000 \
  --planner-max-segments 1200 \
  --planner-retry-cap 4200000 \
  --per-level-timeout-sec 720 \
  --save rl/models/strict_timeout_levels1_30_flashmove_v3_all30.gif \
  --manifest-out rl/models/strict_timeout_levels1_30_flashmove_v3_all30_manifest.json
```

## Key Files

- `rl/whg_full_env.py`: strict full-game environment
- `rl/full_planner.py`: time-aware + hard-level fallback planner
- `rl/render_strict_timeout_sweep.py`: full sweep runner + GIF generator
- `rl/models/strict_timeout_levels1_30_flashmove_v3_all30.gif`: all-30 strict gameplay
- `rl/models/strict_timeout_levels1_30_flashmove_v3_all30_manifest.json`: per-level verification summary

## Credits

- Original game: SnubbyLand / Armor Games
- Java recreation base: Dan Convey
- TinySound library: finnkuusisto
