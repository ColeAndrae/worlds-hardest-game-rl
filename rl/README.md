# RL Module

This folder contains the full reinforcement learning and planning stack for the strict 30-level pipeline.

## Main Components

- `whg_full_env.py`: strict full-game environment (movement, collisions, checkpoints, coins)
- `full_planner.py`: time-aware planner + fallback search for hard levels
- `extract_flash_levels.py`: builds dataset from SWF/XFL export
- `train_full_agent.py`: per-level training runner
- `watch_full_agent.py`: rollout renderer
- `render_strict_timeout_sweep.py`: strict batch sweep with manifest output
- `render_strict_best_effort.py`: best-effort strict sweep runner

## Dataset Build

```bash
python3 rl/extract_flash_levels.py \
  --xfl flash_xfl/the-worlds-hardest-g-1043817f/DOMDocument.xml \
  --decompiled-root /Users/cole/Desktop/worldshardestgame_decompiled \
  --out-dir rl/data/flash_levels
```

## Strict Full Sweep

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

## Legacy Level-1 Stack

- `whg_env.py`
- `planner.py`
- `train_agent.py`
- `watch_agent.py`

```bash
python3 rl/train_agent.py
python3 rl/watch_agent.py --model rl/models/level1_qtable.npz
```
