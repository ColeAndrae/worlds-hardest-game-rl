from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle, Rectangle

from whg_env import WorldsHardestGameEnv

State = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch a trained level-1 RL policy.")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "level1_qtable.npz",
        help="Path to saved model file.",
    )
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS.")
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save GIF (example: rl/models/level1_run.gif).",
    )
    parser.add_argument("--no-gui", action="store_true", help="Do not open an interactive window.")
    parser.add_argument("--max-macro-steps", type=int, default=260, help="Maximum decisions in rollout.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame in playback/output (higher = faster, smaller file).",
    )
    return parser.parse_args()


def load_model(path: Path) -> Tuple[Dict[State, np.ndarray], Dict[str, object], List[int]]:
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    data = np.load(path, allow_pickle=False)

    states = data["states"]
    q_values = data["q_values"]
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}

    q_table: Dict[State, np.ndarray] = {}
    for state_arr, q_arr in zip(states, q_values):
        state = (int(state_arr[0]), int(state_arr[1]), int(state_arr[2]))
        q_table[state] = q_arr.astype(np.float32, copy=False)

    expert_actions: List[int] = []
    if "expert_actions" in data:
        expert_actions = [int(v) for v in data["expert_actions"].tolist()]

    return q_table, metadata, expert_actions


def greedy_action(q_values: np.ndarray) -> int:
    return int(np.argmax(q_values))


def rollout(
    env: WorldsHardestGameEnv,
    q_table: Dict[State, np.ndarray],
    max_macro_steps: int,
) -> Tuple[List[Dict[str, object]], str, int]:
    state = env.reset()
    frames: List[Dict[str, object]] = [env.snapshot()]
    result = "timeout"

    for _ in range(max_macro_steps):
        q_values = q_table.get(state)
        action = greedy_action(q_values) if q_values is not None else WorldsHardestGameEnv.ACTION_STAY

        state, _, done, info = env.step(action, capture_frames=True)
        frames.extend(info["frames"])

        if done:
            result = str(info["result"])
            break

    return frames, result, env.total_micro_steps


def rollout_expert(
    env: WorldsHardestGameEnv,
    expert_actions: List[int],
) -> Tuple[List[Dict[str, object]], str, int]:
    env.reset()
    frames: List[Dict[str, object]] = [env.snapshot()]
    result = "timeout"

    for action in expert_actions:
        _, _, done, info = env.step(action, capture_frames=True)
        frames.extend(info["frames"])
        if done:
            result = str(info["result"])
            break

    return frames, result, env.total_micro_steps


def apply_frame_stride(frames: List[Dict[str, object]], frame_stride: int) -> List[Dict[str, object]]:
    if frame_stride <= 1 or len(frames) <= 2:
        return frames

    reduced = frames[::frame_stride]
    if reduced[-1] is not frames[-1]:
        reduced.append(frames[-1])
    return reduced


def tile_color(tile_type: int, x: int, y: int):
    if tile_type == 0:
        return None
    if tile_type == 1:
        even_x = x % 2 == 0
        even_y = y % 2 == 0
        if even_x == even_y:
            return (230 / 255.0, 230 / 255.0, 1.0)
        return (1.0, 1.0, 1.0)
    return (181 / 255.0, 254 / 255.0, 180 / 255.0)


def render_animation(
    env: WorldsHardestGameEnv,
    frames: List[Dict[str, object]],
    result: str,
    fps: int,
    save_path: Path | None,
    no_gui: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 7.5))

    ax.set_xlim(0, env.BOARD_WIDTH)
    ax.set_ylim(env.BOARD_HEIGHT, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    background = Rectangle((0, 0), env.BOARD_WIDTH, env.BOARD_HEIGHT, facecolor=(180 / 255.0, 181 / 255.0, 254 / 255.0))
    ax.add_patch(background)

    for y, row in enumerate(env.tile_types):
        for x, tile in enumerate(row):
            color = tile_color(tile, x, y)
            if color is None:
                continue
            rect = Rectangle((x * env.TILE_SIZE, y * env.TILE_SIZE), env.TILE_SIZE, env.TILE_SIZE, facecolor=color, edgecolor="none")
            ax.add_patch(rect)

    dot_patches: List[Circle] = []
    for _ in env.dot_specs:
        patch = Circle((0.0, 0.0), radius=8.0, facecolor=(0 / 255.0, 72 / 255.0, 255 / 255.0), edgecolor="black", linewidth=2.0)
        dot_patches.append(patch)
        ax.add_patch(patch)

    player_outer = Rectangle((0.0, 0.0), 31.0, 31.0, facecolor="black", edgecolor="none")
    player_inner = Rectangle((0.0, 0.0), 25.0, 25.0, facecolor=(1.0, 0.0, 0.0), edgecolor="none")
    ax.add_patch(player_outer)
    ax.add_patch(player_inner)

    status_text = ax.text(
        8,
        18,
        "",
        color="black",
        fontsize=12,
        fontweight="bold",
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none", "pad": 2},
    )

    def update(frame_idx: int):
        frame = frames[frame_idx]
        px, py = frame["player"]
        player_outer.set_xy((px - 15.0, py - 15.0))
        player_inner.set_xy((px - 12.0, py - 12.0))

        dot_positions = frame["dots"]
        for patch, (dot_x, dot_y) in zip(dot_patches, dot_positions):
            patch.center = (dot_x + 20.0, dot_y + 20.0)

        status_text.set_text(
            f"result={result}  frame={frame_idx + 1}/{len(frames)}  "
            f"macro={frame['macro_steps']}  micro={frame['micro_steps']}"
        )

        artists = [player_outer, player_inner, status_text]
        artists.extend(dot_patches)
        return artists

    interval_ms = max(1, int(1000 / max(1, fps)))
    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=interval_ms,
        blit=True,
        repeat=False,
    )

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        writer = PillowWriter(fps=fps)
        anim.save(str(save_path), writer=writer)
        print(f"saved animation: {save_path}")

    if no_gui:
        plt.close(fig)
    else:
        plt.show()


def main() -> None:
    args = parse_args()
    q_table, metadata, expert_actions = load_model(args.model)

    action_repeat = int(metadata.get("action_repeat", 4))
    env = WorldsHardestGameEnv(action_repeat=action_repeat, max_macro_steps=args.max_macro_steps)

    frames, result, micro_steps = rollout(env, q_table, args.max_macro_steps)
    print(f"policy rollout result={result} micro_steps={micro_steps} frames={len(frames)}")

    if result != "win" and expert_actions:
        print("policy did not win; falling back to expert path from the model for visualization")
        env = WorldsHardestGameEnv(action_repeat=action_repeat, max_macro_steps=args.max_macro_steps)
        frames, result, micro_steps = rollout_expert(env, expert_actions)
        print(f"expert rollout result={result} micro_steps={micro_steps} frames={len(frames)}")

    frames = apply_frame_stride(frames, args.frame_stride)
    if args.frame_stride > 1:
        print(f"applied frame_stride={args.frame_stride}; playback_frames={len(frames)}")

    render_animation(env, frames, result, args.fps, args.save, args.no_gui)


if __name__ == "__main__":
    main()
