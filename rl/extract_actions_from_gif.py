from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from watch_full_agent import build_level_background
from whg_full_env import WorldsHardestGameFullEnv


ACTIONS_BY_SIGN: Dict[Tuple[int, int], int] = {
    (0, 0): 0,
    (0, -1): 1,
    (0, 1): 2,
    (-1, 0): 3,
    (1, 0): 4,
    (-1, -1): 5,
    (1, -1): 6,
    (-1, 1): 7,
    (1, 1): 8,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract per-level action traces from a rendered WHG GIF.")
    parser.add_argument("--gif", type=Path, required=True, help="Input GIF (typically strict levels 1-10 rollout).")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "flash_levels",
        help="Extracted flash dataset directory.",
    )
    parser.add_argument("--levels", type=str, default="1-10", help="Level range to recover, e.g. 1-10.")
    parser.add_argument("--enemy-hit-radius", type=float, default=6.0, help="Strict enemy radius for replay checks.")
    parser.add_argument("--out-json", type=Path, required=True, help="Output JSON containing extracted actions.")
    return parser.parse_args()


def parse_levels(spec: str) -> List[int]:
    out: List[int] = []
    for chunk in spec.split(","):
        part = chunk.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = int(lo_s)
            hi = int(hi_s)
            if lo > hi:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    levels = sorted(set(out))
    if not levels:
        raise ValueError("No levels parsed from --levels")
    return levels


def load_frames(path: Path) -> List[np.ndarray]:
    im = Image.open(path)
    frames: List[np.ndarray] = []
    idx = 0
    while True:
        try:
            im.seek(idx)
        except EOFError:
            break
        frames.append(np.array(im.convert("RGB"), dtype=np.uint8))
        idx += 1
    return frames


def sign(v: float, eps: float = 0.5) -> int:
    if v < -eps:
        return -1
    if v > eps:
        return 1
    return 0


def detect_player_center(arr: np.ndarray) -> Tuple[float, float] | None:
    # Player inner square is pure red.
    mask = (arr[:, :, 0] > 240) & (arr[:, :, 1] < 40) & (arr[:, :, 2] < 40)
    ys, xs = np.where(mask)
    if xs.size <= 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def frame_level_pred(arr: np.ndarray, bg_arrays: Dict[int, np.ndarray]) -> int:
    # Ignore top HUD band where text changes every frame.
    crop = arr[30:, :, :].astype(np.int16, copy=False)
    best_level = list(bg_arrays.keys())[0]
    best_score = None
    for level, bg in bg_arrays.items():
        d = np.abs(crop - bg[30:, :, :].astype(np.int16, copy=False)).mean()
        if best_score is None or d < best_score:
            best_score = float(d)
            best_level = int(level)
    return best_level


def monotonic_labels(raw_preds: List[int], ordered_levels: List[int], streak_required: int = 3) -> List[int]:
    idx = 0
    streak = 0
    out: List[int] = []
    for p in raw_preds:
        cur = ordered_levels[idx]
        nxt = ordered_levels[idx + 1] if idx + 1 < len(ordered_levels) else None
        if nxt is not None and p == nxt:
            streak += 1
            if streak >= streak_required:
                idx += 1
                streak = 0
        else:
            if p == cur:
                streak = 0
        out.append(ordered_levels[idx])
    return out


def extract_actions(
    frames: List[np.ndarray],
    level_labels: List[int],
    ordered_levels: List[int],
) -> Dict[int, List[int]]:
    per_level_centers: Dict[int, List[Tuple[float, float]]] = {lvl: [] for lvl in ordered_levels}
    for fr, lvl in zip(frames, level_labels):
        c = detect_player_center(fr)
        if c is not None:
            per_level_centers[int(lvl)].append(c)

    per_level_actions: Dict[int, List[int]] = {}
    for lvl in ordered_levels:
        centers = per_level_centers[lvl]
        actions: List[int] = []
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i - 1][0]
            dy = centers[i][1] - centers[i - 1][1]
            sx = sign(dx)
            sy = sign(dy)
            actions.append(ACTIONS_BY_SIGN[(sx, sy)])
        per_level_actions[lvl] = actions
    return per_level_actions


def validate_actions(
    dataset_dir: Path,
    levels: List[int],
    actions_by_level: Dict[int, List[int]],
    enemy_hit_radius: float,
) -> Dict[int, Dict[str, object]]:
    results: Dict[int, Dict[str, object]] = {}
    for lvl in levels:
        env = WorldsHardestGameFullEnv(
            dataset_dir=dataset_dir,
            level_num=int(lvl),
            action_repeat=1,
            max_macro_steps=12_000,
            terminate_on_death=True,
            enemy_hit_radius=float(enemy_hit_radius),
            ignore_enemies=False,
        )
        env.reset()
        result = "timeout"
        for action in actions_by_level.get(lvl, []):
            _, _, done, info = env.step(int(action), capture_frames=False)
            if done:
                result = str(info["result"])
                break
        results[lvl] = {
            "actions": int(len(actions_by_level.get(lvl, []))),
            "result": result,
            "macro_steps": int(env.macro_steps),
        }
    return results


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels)
    frames = load_frames(args.gif)
    if not frames:
        raise RuntimeError(f"No frames in GIF: {args.gif}")

    bg_arrays: Dict[int, np.ndarray] = {}
    for lvl in levels:
        env = WorldsHardestGameFullEnv(
            dataset_dir=args.dataset_dir,
            level_num=int(lvl),
            enemy_hit_radius=float(args.enemy_hit_radius),
            ignore_enemies=False,
        )
        bg = build_level_background(env, scale=1)
        bg_arrays[lvl] = np.array(bg.convert("RGB"), dtype=np.uint8)

    raw_preds = [frame_level_pred(fr, bg_arrays) for fr in frames]
    labels = monotonic_labels(raw_preds, levels, streak_required=3)
    actions = extract_actions(frames, labels, levels)
    validation = validate_actions(args.dataset_dir, levels, actions, args.enemy_hit_radius)

    payload = {
        "gif": str(args.gif),
        "frames": int(len(frames)),
        "levels": levels,
        "results": {str(k): v for k, v in validation.items()},
        "actions": {str(k): v for k, v in actions.items()},
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload), encoding="utf-8")

    print(json.dumps(payload["results"], indent=2))
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
