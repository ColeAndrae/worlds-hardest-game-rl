from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import time
from pathlib import Path
from typing import Dict, List

from PIL import Image

from full_planner import plan_level_greedy
from watch_full_agent import apply_frame_stride, build_level_background, render_frame, rollout_actions
from whg_full_env import WorldsHardestGameFullEnv


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Strict planner sweep with per-level timeout isolation.")
    parser.add_argument("--levels", type=str, default="1-30", help="Level set, e.g. '1-10' or '1,2,5-8'.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "flash_levels",
        help="Extracted flash dataset directory.",
    )
    parser.add_argument("--enemy-hit-radius", type=float, default=6.0, help="Strict collision radius.")
    parser.add_argument("--max-macro-steps", type=int, default=16_000, help="Per-level macro-step cap.")
    parser.add_argument("--planner-max-expand", type=int, default=350_000, help="Planner expansions per segment.")
    parser.add_argument("--planner-max-segments", type=int, default=450, help="Planner segment cap.")
    parser.add_argument("--planner-retry-cap", type=int, default=1_200_000, help="Planner retry expansion cap.")
    parser.add_argument("--per-level-timeout-sec", type=int, default=180, help="Wall-clock timeout per level.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Keep every Nth frame.")
    parser.add_argument("--fps", type=int, default=30, help="Output GIF fps.")
    parser.add_argument("--scale", type=int, default=1, help="Render scale.")
    parser.add_argument(
        "--save",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "strict_timeout_sweep.gif",
        help="Output GIF path.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "strict_timeout_sweep_manifest.json",
        help="JSON summary path.",
    )
    return parser.parse_args()


def _solve_level_worker(
    out_q: mp.Queue,
    dataset_dir: str,
    level: int,
    enemy_hit_radius: float,
    max_macro_steps: int,
    planner_max_expand: int,
    planner_max_segments: int,
    planner_retry_cap: int,
) -> None:
    try:
        t0 = time.time()
        env_plan = WorldsHardestGameFullEnv(
            dataset_dir=dataset_dir,
            level_num=int(level),
            action_repeat=1,
            max_macro_steps=int(max_macro_steps),
            terminate_on_death=True,
            enemy_hit_radius=float(enemy_hit_radius),
            ignore_enemies=False,
        )
        plan = plan_level_greedy(
            env_plan,
            max_expand_per_segment=int(planner_max_expand),
            max_segments=int(planner_max_segments),
            allow_wait=True,
            verbose=False,
            retry_expand_cap=int(planner_retry_cap),
        )
        result = "no_plan"
        if plan.found and plan.actions:
            env_roll = WorldsHardestGameFullEnv(
                dataset_dir=dataset_dir,
                level_num=int(level),
                action_repeat=1,
                max_macro_steps=int(max_macro_steps),
                terminate_on_death=True,
                enemy_hit_radius=float(enemy_hit_radius),
                ignore_enemies=False,
            )
            _, result = rollout_actions(env_roll, [int(a) for a in plan.actions])

        out_q.put(
            {
                "ok": True,
                "level": int(level),
                "plan_found": bool(plan.found),
                "plan_reason": str(plan.reason),
                "plan_actions": int(len(plan.actions)),
                "plan_expanded": int(plan.expanded),
                "result": str(result),
                "elapsed_sec": round(float(time.time() - t0), 2),
                "actions": [int(a) for a in plan.actions] if bool(plan.found and result == "win") else [],
            }
        )
    except Exception as exc:  # pragma: no cover - best effort worker safety
        out_q.put(
            {
                "ok": False,
                "level": int(level),
                "error": repr(exc),
                "plan_found": False,
                "plan_reason": "worker_error",
                "plan_actions": 0,
                "plan_expanded": 0,
                "result": "error",
                "elapsed_sec": 0.0,
                "actions": [],
            }
        )


def solve_level_with_timeout(
    dataset_dir: Path,
    level: int,
    enemy_hit_radius: float,
    max_macro_steps: int,
    planner_max_expand: int,
    planner_max_segments: int,
    planner_retry_cap: int,
    timeout_sec: int,
) -> Dict[str, object]:
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_solve_level_worker,
        args=(
            out_q,
            str(dataset_dir),
            int(level),
            float(enemy_hit_radius),
            int(max_macro_steps),
            int(planner_max_expand),
            int(planner_max_segments),
            int(planner_retry_cap),
        ),
    )
    proc.start()
    proc.join(timeout=max(1, int(timeout_sec)))
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5)
        return {
            "ok": False,
            "level": int(level),
            "error": "timeout",
            "plan_found": False,
            "plan_reason": "timeout",
            "plan_actions": 0,
            "plan_expanded": 0,
            "result": "timeout",
            "elapsed_sec": float(timeout_sec),
            "actions": [],
        }
    try:
        return out_q.get_nowait()
    except queue.Empty:
        return {
            "ok": False,
            "level": int(level),
            "error": "empty_result",
            "plan_found": False,
            "plan_reason": "empty_result",
            "plan_actions": 0,
            "plan_expanded": 0,
            "result": "error",
            "elapsed_sec": 0.0,
            "actions": [],
        }


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels)

    summaries: List[Dict[str, object]] = []
    kept_levels: List[int] = []
    all_frames: List[Dict[str, object]] = []

    for lvl in levels:
        rec = solve_level_with_timeout(
            dataset_dir=args.dataset_dir,
            level=int(lvl),
            enemy_hit_radius=float(args.enemy_hit_radius),
            max_macro_steps=int(args.max_macro_steps),
            planner_max_expand=int(args.planner_max_expand),
            planner_max_segments=int(args.planner_max_segments),
            planner_retry_cap=int(args.planner_retry_cap),
            timeout_sec=int(args.per_level_timeout_sec),
        )
        keep = bool(rec.get("plan_found")) and str(rec.get("result")) == "win" and bool(rec.get("actions"))
        rec["kept"] = bool(keep)
        summaries.append(rec)
        print(
            f"level {lvl:02d}: plan_found={rec.get('plan_found')} reason={rec.get('plan_reason')} "
            f"result={rec.get('result')} elapsed={rec.get('elapsed_sec')} kept={keep}"
        )
        if not keep:
            continue

        actions = [int(a) for a in rec.get("actions", [])]
        env_roll = WorldsHardestGameFullEnv(
            dataset_dir=args.dataset_dir,
            level_num=int(lvl),
            action_repeat=1,
            max_macro_steps=int(args.max_macro_steps),
            terminate_on_death=True,
            enemy_hit_radius=float(args.enemy_hit_radius),
            ignore_enemies=False,
        )
        frames, result = rollout_actions(env_roll, actions)
        if result == "win":
            kept_levels.append(int(lvl))
            all_frames.extend(frames)
        else:
            rec["kept"] = False
            rec["result"] = str(result)

    if not all_frames:
        raise RuntimeError("No winning rollouts captured.")

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
                max_macro_steps=int(args.max_macro_steps),
                terminate_on_death=True,
                enemy_hit_radius=float(args.enemy_hit_radius),
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
        "levels_kept": kept_levels,
        "settings": {
            "enemy_hit_radius": float(args.enemy_hit_radius),
            "max_macro_steps": int(args.max_macro_steps),
            "planner_max_expand": int(args.planner_max_expand),
            "planner_max_segments": int(args.planner_max_segments),
            "planner_retry_cap": int(args.planner_retry_cap),
            "per_level_timeout_sec": int(args.per_level_timeout_sec),
            "frame_stride": int(args.frame_stride),
            "fps": int(args.fps),
            "scale": int(args.scale),
        },
        "summary": summaries,
        "output_frames": int(len(rendered)),
        "output_gif": str(args.save),
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"saved animation: {args.save}")
    print(f"wrote manifest: {args.manifest_out}")
    print(f"kept levels: {kept_levels}")


if __name__ == "__main__":
    main()
