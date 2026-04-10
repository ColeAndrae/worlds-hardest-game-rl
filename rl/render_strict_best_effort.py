from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from PIL import Image

from full_planner import plan_level_greedy
from watch_full_agent import apply_frame_stride, build_level_background, render_frame, rollout_actions
from whg_full_env import WorldsHardestGameFullEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render strict enemy-aware wins for as many levels as possible (best-effort)."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "flash_levels",
        help="Extracted flash dataset directory.",
    )
    parser.add_argument(
        "--sweep-json",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "strict_sweep_45s.json",
        help="Sweep JSON used to select already-verified strict wins.",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="",
        help="Optional explicit levels (e.g. '1,2,4-6'). Empty means use wins from --sweep-json.",
    )
    parser.add_argument("--enemy-hit-radius", type=float, default=6.0, help="Strict collision radius.")
    parser.add_argument("--max-macro-steps", type=int, default=12_000, help="Per-level macro-step cap.")
    parser.add_argument("--planner-max-expand", type=int, default=350_000, help="Planner expansions per segment.")
    parser.add_argument("--planner-max-segments", type=int, default=500, help="Planner segment cap.")
    parser.add_argument("--planner-retry-cap", type=int, default=1_200_000, help="Planner retry expansion cap.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--fps", type=int, default=30, help="GIF fps.")
    parser.add_argument("--scale", type=int, default=1, help="Render scale.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "strict_best_effort.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "strict_best_effort_manifest.json",
        help="JSON summary path.",
    )
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
    for lvl in levels:
        if lvl < 1 or lvl > 30:
            raise ValueError(f"Invalid level: {lvl}")
    return levels


def levels_from_sweep(path: Path) -> List[int]:
    if not path.exists():
        raise FileNotFoundError(f"Sweep JSON not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: List[int] = []
    for key, entry in payload.get("levels", {}).items():
        if bool(entry.get("plan_found")) and str(entry.get("replay_result")) == "win":
            out.append(int(key))
    return sorted(set(out))


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels) if args.levels.strip() else levels_from_sweep(args.sweep_json)
    if not levels:
        raise RuntimeError("No levels selected to render.")

    all_frames: List[Dict[str, object]] = []
    summary: List[Dict[str, object]] = []

    for lvl in levels:
        env_plan = WorldsHardestGameFullEnv(
            dataset_dir=args.dataset_dir,
            level_num=lvl,
            action_repeat=1,
            max_macro_steps=args.max_macro_steps,
            terminate_on_death=True,
            enemy_hit_radius=args.enemy_hit_radius,
            ignore_enemies=False,
        )
        plan = plan_level_greedy(
            env_plan,
            max_expand_per_segment=int(args.planner_max_expand),
            max_segments=int(args.planner_max_segments),
            allow_wait=True,
            verbose=False,
            retry_expand_cap=int(args.planner_retry_cap),
        )

        result = "no_plan"
        level_frames: List[Dict[str, object]] = []
        if plan.found and plan.actions:
            env_roll = WorldsHardestGameFullEnv(
                dataset_dir=args.dataset_dir,
                level_num=lvl,
                action_repeat=1,
                max_macro_steps=args.max_macro_steps,
                terminate_on_death=True,
                enemy_hit_radius=args.enemy_hit_radius,
                ignore_enemies=False,
            )
            level_frames, result = rollout_actions(env_roll, [int(a) for a in plan.actions])

        ok = bool(plan.found and result == "win")
        summary.append(
            {
                "level": int(lvl),
                "plan_found": bool(plan.found),
                "plan_reason": str(plan.reason),
                "plan_actions": int(len(plan.actions)),
                "plan_expanded": int(plan.expanded),
                "result": result,
                "frames": int(len(level_frames)),
                "kept": bool(ok),
            }
        )
        print(
            f"level {lvl:02d}: plan_found={plan.found} reason={plan.reason} "
            f"result={result} frames={len(level_frames)} kept={ok}"
        )
        if ok:
            all_frames.extend(level_frames)

    if not all_frames:
        raise RuntimeError("No winning level rollouts available to render.")

    reduced = apply_frame_stride(all_frames, max(1, int(args.frame_stride)))
    backgrounds: Dict[int, Image.Image] = {}
    rendered: List[Image.Image] = []
    for frame in reduced:
        lvl = int(frame["level_num"])
        if lvl not in backgrounds:
            env_bg = WorldsHardestGameFullEnv(
                dataset_dir=args.dataset_dir,
                level_num=lvl,
                action_repeat=1,
                max_macro_steps=args.max_macro_steps,
                terminate_on_death=True,
                enemy_hit_radius=args.enemy_hit_radius,
                ignore_enemies=False,
            )
            backgrounds[lvl] = build_level_background(env_bg, scale=max(1, int(args.scale)))
        rendered.append(render_frame(backgrounds[lvl], frame, scale=max(1, int(args.scale))))

    args.save.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = max(1, int(1000 / max(1, int(args.fps))))
    rendered[0].save(
        args.save,
        save_all=True,
        append_images=rendered[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )

    manifest = {
        "levels_input": levels,
        "levels_kept": [int(s["level"]) for s in summary if bool(s["kept"])],
        "settings": {
            "enemy_hit_radius": float(args.enemy_hit_radius),
            "max_macro_steps": int(args.max_macro_steps),
            "planner_max_expand": int(args.planner_max_expand),
            "planner_max_segments": int(args.planner_max_segments),
            "planner_retry_cap": int(args.planner_retry_cap),
            "frame_stride": int(args.frame_stride),
            "fps": int(args.fps),
            "scale": int(args.scale),
        },
        "summary": summary,
        "output_frames": int(len(rendered)),
        "output_gif": str(args.save),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"saved animation: {args.save}")
    print(f"wrote manifest: {args.manifest_out}")
    print(f"kept levels: {manifest['levels_kept']}")


if __name__ == "__main__":
    main()
