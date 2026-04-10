from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from full_planner import plan_level_greedy
from whg_full_env import State, WorldsHardestGameFullEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch/save full-game WHG RL rollouts.")
    parser.add_argument("--levels", type=str, default="1-30", help="Level set, e.g. '1-30' or '1,2,5-8'.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "flash_levels",
        help="Extracted flash dataset directory.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "full",
        help="Directory containing level_XX_qtable.npz models.",
    )
    parser.add_argument("--action-repeat", type=int, default=1, help="Micro-steps per decision.")
    parser.add_argument("--max-macro-steps", type=int, default=1400, help="Max macro-steps per level rollout.")
    parser.add_argument(
        "--enemy-hit-radius",
        type=float,
        default=6.0,
        help="Enemy collision radius in level pixels (strict AABB-vs-circle overlap).",
    )
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS for GIF/preview.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=2,
        help="Keep every Nth frame in output (higher = faster, smaller GIF).",
    )
    parser.add_argument("--scale", type=int, default=2, help="Output pixel scale multiplier.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "full_run.gif",
        help="GIF output path.",
    )
    parser.add_argument(
        "--planner-fallback",
        action="store_true",
        help="If model policy fails, fallback to level expert plan computed on the fly.",
    )
    parser.add_argument(
        "--planner-max-expand",
        type=int,
        default=2_200_000,
        help="Maximum planner node expansions per segment when fallback is enabled.",
    )
    parser.add_argument(
        "--planner-max-segments",
        type=int,
        default=260,
        help="Maximum planner segments per level when fallback is enabled.",
    )
    parser.add_argument("--no-gui", action="store_true", help="Keep for compatibility (GIF rendering is default).")
    parser.add_argument(
        "--ignore-enemies",
        action="store_true",
        help="Disable enemy collisions for guaranteed all-level completion demos.",
    )
    parser.add_argument(
        "--allow-non-strict",
        action="store_true",
        help="Required with --ignore-enemies to intentionally run non-strict collision demos.",
    )
    return parser.parse_args()


def parse_levels(spec: str) -> List[int]:
    out: List[int] = []
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            a_s, b_s = part.split("-", 1)
            a = int(a_s)
            b = int(b_s)
            if a > b:
                a, b = b, a
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    levels = sorted(set(out))
    for lvl in levels:
        if lvl < 1 or lvl > 30:
            raise ValueError(f"Invalid level: {lvl}")
    return levels


def load_model(path: Path) -> Tuple[Dict[State, np.ndarray], Dict[str, object], List[int]]:
    if not path.exists():
        return {}, {}, []
    data = np.load(path, allow_pickle=False)
    states = data["states"]
    q_values = data["q_values"]
    metadata = json.loads(str(data["metadata"])) if "metadata" in data else {}
    q_table: Dict[State, np.ndarray] = {}
    for s, q in zip(states, q_values):
        key: State = (int(s[0]), int(s[1]), int(s[2]), int(s[3]), int(s[4]))
        q_table[key] = q.astype(np.float32, copy=False)
    expert_actions: List[int] = []
    if "expert_actions" in data:
        expert_actions = [int(v) for v in data["expert_actions"].tolist()]
    return q_table, metadata, expert_actions


def greedy_action(q_values: np.ndarray) -> int:
    return int(np.argmax(q_values))


def rollout_policy(
    env: WorldsHardestGameFullEnv,
    q_table: Dict[State, np.ndarray],
    max_macro_steps: int,
) -> Tuple[List[Dict[str, object]], str]:
    state = env.reset()
    frames: List[Dict[str, object]] = [env.snapshot()]
    result = "timeout"
    for _ in range(max_macro_steps):
        q_values = q_table.get(state)
        action = greedy_action(q_values) if q_values is not None else WorldsHardestGameFullEnv.ACTION_STAY
        state, _, done, info = env.step(action, capture_frames=True)
        frames.extend(info["frames"])
        if done:
            result = str(info["result"])
            break
    return frames, result


def rollout_actions(
    env: WorldsHardestGameFullEnv,
    actions: List[int],
) -> Tuple[List[Dict[str, object]], str]:
    env.reset()
    frames: List[Dict[str, object]] = [env.snapshot()]
    result = "timeout"
    for action in actions:
        _, _, done, info = env.step(action, capture_frames=True)
        frames.extend(info["frames"])
        if done:
            result = str(info["result"])
            break
    return frames, result


def apply_frame_stride(frames: List[Dict[str, object]], frame_stride: int) -> List[Dict[str, object]]:
    if frame_stride <= 1 or len(frames) <= 2:
        return frames
    reduced = frames[::frame_stride]
    if reduced[-1] is not frames[-1]:
        reduced.append(frames[-1])
    return reduced


def build_level_background(env: WorldsHardestGameFullEnv, scale: int) -> Image.Image:
    h = env.height
    w = env.width
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, :, :] = np.array([180, 181, 254], dtype=np.uint8)

    yy, xx = np.indices((h, w))
    checker = ((xx // 25) + (yy // 25)) % 2 == 0
    walk = env.walkable_mask
    arr[walk & checker] = np.array([248, 248, 255], dtype=np.uint8)
    arr[walk & (~checker)] = np.array([255, 255, 255], dtype=np.uint8)

    base = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(base)
    for region in env.check_regions.values():
        draw.rectangle([region.x0, region.y0, region.x1, region.y1], fill=(181, 254, 180), outline=None)

    if scale > 1:
        base = base.resize((w * scale, h * scale), resample=Image.Resampling.NEAREST)
    return base


def render_frame(
    base: Image.Image,
    frame: Dict[str, object],
    scale: int,
) -> Image.Image:
    img = base.copy()
    draw = ImageDraw.Draw(img)

    sx = float(scale)

    coins = frame["coins"]
    for cx, cy, collected in coins:
        if collected:
            continue
        x = float(cx) * sx
        y = float(cy) * sx
        r = 5.0 * sx
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(255, 234, 0), outline=(0, 0, 0), width=max(1, int(sx)))

    enemies = frame["enemies"]
    enemy_hit_radius = float(frame.get("enemy_hit_radius", 7.0))
    for ex, ey in enemies:
        x = float(ex) * sx
        y = float(ey) * sx
        r = enemy_hit_radius * sx
        draw.ellipse([x - r, y - r, x + r, y + r], fill=(0, 72, 255), outline=(0, 0, 0), width=max(1, int(sx)))

    px, py = frame["player"]
    px *= sx
    py *= sx
    outer = 9.0 * sx
    inner = 7.0 * sx
    draw.rectangle([px - outer, py - outer, px + outer, py + outer], fill=(0, 0, 0))
    draw.rectangle([px - inner, py - inner, px + inner, py + inner], fill=(255, 0, 0))

    level_num = int(frame["level_num"])
    result = str(frame["result"])
    macro_steps = int(frame["macro_steps"])
    coins_collected = int(frame["coins_collected"])
    coins_required = int(frame["coins_required"])
    deaths = int(frame["deaths"])
    ignore_enemies = bool(frame.get("ignore_enemies", False))
    mode = "no-enemy-eval" if ignore_enemies else "strict"

    text = (
        f"Level {level_num:02d}  mode={mode}  result={result}  macro={macro_steps}  "
        f"coins={coins_collected}/{coins_required}  deaths={deaths}"
    )
    draw.rectangle([4, 4, min(img.size[0] - 4, 4 + int(9.5 * len(text))), 24], fill=(255, 255, 255))
    draw.text((8, 8), text, fill=(0, 0, 0))
    return img


def main() -> None:
    args = parse_args()
    if args.ignore_enemies and not args.allow_non_strict:
        raise ValueError(
            "--ignore-enemies is non-strict mode and allows passing through blue enemies. "
            "Re-run with --allow-non-strict to confirm this is intentional."
        )
    levels = parse_levels(args.levels)
    all_frames: List[Dict[str, object]] = []
    level_summaries: List[str] = []

    for level in levels:
        env = WorldsHardestGameFullEnv(
            dataset_dir=args.dataset_dir,
            level_num=level,
            action_repeat=args.action_repeat,
            max_macro_steps=args.max_macro_steps,
            terminate_on_death=True,
            enemy_hit_radius=args.enemy_hit_radius,
            ignore_enemies=args.ignore_enemies,
        )

        model_path = args.model_dir / f"level_{level:02d}_qtable.npz"
        q_table, metadata, expert_actions = load_model(model_path)

        if q_table:
            frames, result = rollout_policy(env, q_table, args.max_macro_steps)
        else:
            frames = [env.snapshot()]
            result = "no_model"
        used_fallback = False

        if result != "win":
            fallback_actions = expert_actions
            if not fallback_actions and args.planner_fallback:
                planner_env = WorldsHardestGameFullEnv(
                    dataset_dir=args.dataset_dir,
                    level_num=level,
                    action_repeat=args.action_repeat,
                    max_macro_steps=args.max_macro_steps,
                    terminate_on_death=True,
                    enemy_hit_radius=args.enemy_hit_radius,
                    ignore_enemies=args.ignore_enemies,
                )
                plan = plan_level_greedy(
                    planner_env,
                    max_expand_per_segment=int(args.planner_max_expand),
                    max_segments=int(args.planner_max_segments),
                    allow_wait=True,
                )
                fallback_actions = plan.actions if plan.found else []

            if fallback_actions:
                used_fallback = True
                env = WorldsHardestGameFullEnv(
                    dataset_dir=args.dataset_dir,
                    level_num=level,
                    action_repeat=args.action_repeat,
                    max_macro_steps=args.max_macro_steps,
                    terminate_on_death=True,
                    enemy_hit_radius=args.enemy_hit_radius,
                    ignore_enemies=args.ignore_enemies,
                )
                frames, result = rollout_actions(env, fallback_actions)

        all_frames.extend(frames)
        level_summaries.append(
            f"level {level:02d}: result={result} "
            f"frames={len(frames)} model={'yes' if metadata else 'no'} fallback={'yes' if used_fallback else 'no'}"
        )
        print(level_summaries[-1])

        if result != "win":
            print(f"stopping after level {level:02d} (result={result})")
            break

    all_frames = apply_frame_stride(all_frames, args.frame_stride)
    if not all_frames:
        raise RuntimeError("No frames to render.")

    backgrounds: Dict[int, Image.Image] = {}
    rendered: List[Image.Image] = []
    for frame in all_frames:
        level_num = int(frame["level_num"])
        if level_num not in backgrounds:
            bg_env = WorldsHardestGameFullEnv(
                dataset_dir=args.dataset_dir,
                level_num=level_num,
                action_repeat=args.action_repeat,
                max_macro_steps=args.max_macro_steps,
                terminate_on_death=True,
                enemy_hit_radius=args.enemy_hit_radius,
                ignore_enemies=args.ignore_enemies,
            )
            backgrounds[level_num] = build_level_background(bg_env, scale=max(1, args.scale))
        rendered.append(render_frame(backgrounds[level_num], frame, scale=max(1, args.scale)))

    args.save.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / max(1, args.fps)))
    rendered[0].save(
        args.save,
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    print(f"saved animation: {args.save}")
    print(f"output frames={len(rendered)} fps={args.fps} frame_stride={args.frame_stride}")


if __name__ == "__main__":
    main()
