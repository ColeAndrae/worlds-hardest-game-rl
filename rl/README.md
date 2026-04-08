# RL Module (Level 1)

This folder adds a deterministic, headless environment for **World's Hardest Game level 1** and a train/watch workflow.

## What's Included

- `whg_env.py`: deterministic simulator extracted from the original game logic (movement, collision, dots, goals).
- `planner.py`: A* expert planner for warm-start trajectories.
- `train_agent.py`: tabular Q-learning trainer with expert warm-start.
- `watch_agent.py`: animate a trained policy (and save a GIF).
- `models/`: saved checkpoints and run artifacts.

## Why This Is RL-Friendly

- No Swing/UI dependency during training.
- Deterministic stepping (`reset` / `step`) for reproducibility.
- Action repeat support (`--action-repeat`) for faster learning.
- Simple serialized model format (`.npz`) for quick reload and evaluation.

## Train

From the repo root:

```bash
python3 rl/train_agent.py
```

Optional faster run:

```bash
python3 rl/train_agent.py --episodes 1500 --eval-every 200
```

Model output defaults to:

- `rl/models/level1_qtable.npz`

## Watch

Interactive window:

```bash
python3 rl/watch_agent.py --model rl/models/level1_qtable.npz
```

Save a GIF (no GUI):

```bash
python3 rl/watch_agent.py --model rl/models/level1_qtable.npz --save rl/models/level1_run.gif --no-gui
```

Generate a faster/smaller GIF:

```bash
python3 rl/watch_agent.py --model rl/models/level1_qtable.npz --save rl/models/level1_run_fast.gif --fps 30 --frame-stride 3 --no-gui
```

## Notes

- If the learned Q policy misses, `watch_agent.py` automatically falls back to the stored expert path from the same model file so you can still visualize a complete winning run.
- The environment currently targets level 1 first, which is ideal for iteration speed and reproducibility.
