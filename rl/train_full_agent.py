from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Tuple

import numpy as np

from full_planner import LevelPlan, plan_level_greedy
from whg_full_env import State, WorldsHardestGameFullEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train level-by-level tabular RL policies for all 30 WHG levels.")
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
        help="Directory for per-level model outputs.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--action-repeat", type=int, default=1, help="Micro-steps per decision.")
    parser.add_argument("--max-macro-steps", type=int, default=1400, help="Per-episode macro-step cap.")
    parser.add_argument("--episodes-per-level", type=int, default=250, help="Q-learning episodes per level.")
    parser.add_argument("--eval-every", type=int, default=100, help="Eval interval.")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Episodes per eval run.")
    parser.add_argument("--alpha", type=float, default=0.18, help="Q-learning learning rate.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=0.25, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=0.03, help="Final epsilon.")
    parser.add_argument("--warmstart-bonus", type=float, default=2.5, help="Expert action Q bonus.")
    parser.add_argument("--planner-max-expand", type=int, default=2_200_000, help="A* expansions per segment.")
    parser.add_argument("--planner-max-segments", type=int, default=260, help="Planner segment cap.")
    parser.add_argument("--ignore-enemies", action="store_true", help="Disable enemy collisions during training/planning.")
    parser.add_argument("--skip-q-learning", action="store_true", help="Only compute/save expert warm-start model.")
    parser.add_argument("--quiet", action="store_true", help="Reduce output.")
    return parser.parse_args()


def parse_levels(spec: str) -> List[int]:
    levels: List[int] = []
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
            levels.extend(range(lo, hi + 1))
        else:
            levels.append(int(part))
    uniq = sorted(set(levels))
    for lvl in uniq:
        if lvl < 1 or lvl > 30:
            raise ValueError(f"Level out of range: {lvl}")
    return uniq


def epsilon_for_episode(episode_idx: int, total_episodes: int, start: float, end: float) -> float:
    if total_episodes <= 1:
        return end
    frac = episode_idx / float(total_episodes - 1)
    return end + (start - end) * (1.0 - frac)


def greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    noise = rng.random(q_values.shape[0]) * 1e-8
    return int(np.argmax(q_values + noise))


def warmstart_q_table(
    env: WorldsHardestGameFullEnv,
    q_table: DefaultDict[State, np.ndarray],
    expert_actions: List[int],
    warmstart_bonus: float,
) -> int:
    state = env.reset()
    applied = 0
    for action in expert_actions:
        q_table[state][action] += warmstart_bonus
        state, _, done, _ = env.step(action, capture_frames=False)
        applied += 1
        if done:
            break
    return applied


def evaluate_policy(
    env: WorldsHardestGameFullEnv,
    q_table: DefaultDict[State, np.ndarray],
    episodes: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    wins = 0
    deaths = 0
    timeouts = 0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        info = {"result": "timeout", "macro_steps": env.max_macro_steps}
        while not done:
            action = greedy_action(q_table[state], rng)
            state, _, done, info = env.step(action, capture_frames=False)

        if info["result"] == "win":
            wins += 1
        elif info["result"] == "death":
            deaths += 1
        else:
            timeouts += 1
        total_steps += int(info["macro_steps"])

    return {
        "win_rate": wins / float(episodes),
        "death_rate": deaths / float(episodes),
        "timeout_rate": timeouts / float(episodes),
        "avg_macro_steps": total_steps / float(episodes),
    }


def save_q_table(
    model_path: Path,
    q_table: DefaultDict[State, np.ndarray],
    metadata: Dict[str, object],
    expert_actions: List[int],
) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    states: List[State] = list(q_table.keys())
    if states:
        state_array = np.array(states, dtype=np.int16)
        q_values = np.array([q_table[s] for s in states], dtype=np.float32)
    else:
        state_array = np.empty((0, 5), dtype=np.int16)
        q_values = np.empty((0, len(WorldsHardestGameFullEnv.ACTIONS)), dtype=np.float32)

    np.savez_compressed(
        model_path,
        states=state_array,
        q_values=q_values,
        metadata=json.dumps(metadata),
        expert_actions=np.array(expert_actions, dtype=np.int16),
    )


def train_level(
    level: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> Dict[str, object]:
    env = WorldsHardestGameFullEnv(
        dataset_dir=args.dataset_dir,
        level_num=level,
        action_repeat=args.action_repeat,
        max_macro_steps=args.max_macro_steps,
        terminate_on_death=True,
        ignore_enemies=args.ignore_enemies,
    )

    plan_env = WorldsHardestGameFullEnv(
        dataset_dir=args.dataset_dir,
        level_num=level,
        action_repeat=args.action_repeat,
        max_macro_steps=args.max_macro_steps,
        terminate_on_death=True,
        ignore_enemies=args.ignore_enemies,
    )
    planner_start = time.time()
    plan: LevelPlan = plan_level_greedy(
        plan_env,
        max_expand_per_segment=args.planner_max_expand,
        max_segments=args.planner_max_segments,
        allow_wait=True,
        verbose=False,
    )
    planner_elapsed = time.time() - planner_start

    q_table: DefaultDict[State, np.ndarray] = defaultdict(
        lambda: np.zeros(len(WorldsHardestGameFullEnv.ACTIONS), dtype=np.float32)
    )
    warm_count = warmstart_q_table(env, q_table, plan.actions, args.warmstart_bonus)

    best_eval = -1.0
    train_start = time.time()
    if not args.skip_q_learning and args.episodes_per_level > 0:
        for episode_idx in range(args.episodes_per_level):
            epsilon = epsilon_for_episode(
                episode_idx, args.episodes_per_level, args.epsilon_start, args.epsilon_end
            )
            state = env.reset()
            done = False

            while not done:
                if rng.random() < epsilon:
                    action = int(rng.integers(0, len(WorldsHardestGameFullEnv.ACTIONS)))
                else:
                    action = greedy_action(q_table[state], rng)

                next_state, reward, done, _ = env.step(action, capture_frames=False)
                best_next = 0.0 if done else float(np.max(q_table[next_state]))
                target = reward + args.gamma * best_next
                q_table[state][action] += args.alpha * (target - q_table[state][action])
                state = next_state

            if (episode_idx + 1) % args.eval_every == 0 or episode_idx == args.episodes_per_level - 1:
                metrics = evaluate_policy(env, q_table, max(3, args.eval_episodes // 2), rng)
                best_eval = max(best_eval, metrics["win_rate"])
                if not args.quiet:
                    print(
                        f"level={level:02d} ep={episode_idx + 1} "
                        f"eval_win={metrics['win_rate']:.3f} "
                        f"eval_death={metrics['death_rate']:.3f} "
                        f"q_states={len(q_table)}"
                    )

    train_elapsed = time.time() - train_start
    final_metrics = evaluate_policy(env, q_table, args.eval_episodes, rng)
    best_eval = max(best_eval, final_metrics["win_rate"])

    model_path = args.model_dir / f"level_{level:02d}_qtable.npz"
    metadata: Dict[str, object] = {
        "level": level,
        "action_repeat": args.action_repeat,
        "max_macro_steps": args.max_macro_steps,
        "episodes": 0 if args.skip_q_learning else args.episodes_per_level,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "planner_found": plan.found,
        "planner_reason": plan.reason,
        "planner_segments": plan.segments,
        "planner_expanded": plan.expanded,
        "planner_elapsed_sec": planner_elapsed,
        "planner_result": plan.result,
        "ignore_enemies": args.ignore_enemies,
        "warmstart_steps": warm_count,
        "best_eval_win_rate": best_eval,
        "final_eval_win_rate": final_metrics["win_rate"],
        "final_eval_death_rate": final_metrics["death_rate"],
        "final_eval_timeout_rate": final_metrics["timeout_rate"],
        "train_elapsed_sec": train_elapsed,
        "q_states": len(q_table),
    }
    save_q_table(model_path, q_table, metadata, plan.actions)

    if not args.quiet:
        print(
            f"saved {model_path} "
            f"planner_found={plan.found} planner_reason={plan.reason} "
            f"eval_win={final_metrics['win_rate']:.3f}"
        )

    return {
        "level": level,
        "model": str(model_path.name),
        **metadata,
    }


def main() -> None:
    args = parse_args()
    levels = parse_levels(args.levels)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    manifest: Dict[str, object] = {
        "levels": levels,
        "seed": args.seed,
        "generated_at_unix": time.time(),
        "entries": [],
    }

    for level in levels:
        entry = train_level(level, args, rng)
        manifest["entries"].append(entry)  # type: ignore[arg-type]

    manifest_path = args.model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
