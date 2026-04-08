from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import numpy as np

from planner import PlanResult, plan_level_actions
from whg_env import WorldsHardestGameEnv

State = Tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent for WHG level 1.")
    parser.add_argument("--episodes", type=int, default=3000, help="Number of Q-learning episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path(__file__).resolve().parent / "models" / "level1_qtable.npz",
        help="Output model path.",
    )
    parser.add_argument("--action-repeat", type=int, default=4, help="Frames per RL action.")
    parser.add_argument("--max-macro-steps", type=int, default=260, help="Per-episode decision limit.")
    parser.add_argument("--alpha", type=float, default=0.2, help="Q-learning learning rate.")
    parser.add_argument("--gamma", type=float, default=0.995, help="Discount factor.")
    parser.add_argument("--epsilon-start", type=float, default=0.20, help="Initial epsilon.")
    parser.add_argument("--epsilon-end", type=float, default=0.02, help="Final epsilon.")
    parser.add_argument("--warmstart-bonus", type=float, default=3.0, help="Expert action Q-value bonus.")
    parser.add_argument("--eval-every", type=int, default=250, help="Evaluation interval in episodes.")
    parser.add_argument("--eval-episodes", type=int, default=30, help="Evaluation episodes.")
    parser.add_argument("--planner-max-expand", type=int, default=1_200_000, help="A* expansion cap.")
    parser.add_argument("--quiet", action="store_true", help="Reduce log output.")
    return parser.parse_args()


def epsilon_for_episode(episode_idx: int, total_episodes: int, start: float, end: float) -> float:
    if total_episodes <= 1:
        return end
    frac = episode_idx / float(total_episodes - 1)
    return end + (start - end) * (1.0 - frac)


def greedy_action(q_values: np.ndarray, rng: np.random.Generator) -> int:
    # Tiny random tie-breaker keeps argmax deterministic enough while avoiding fixed-order bias.
    noise = rng.random(q_values.shape[0]) * 1e-8
    return int(np.argmax(q_values + noise))


def warmstart_q_table(
    env: WorldsHardestGameEnv,
    q_table: DefaultDict[State, np.ndarray],
    expert_actions: Iterable[int],
    warmstart_bonus: float,
) -> int:
    state = env.reset()
    applied = 0
    for action in expert_actions:
        q_table[state][action] += warmstart_bonus
        state, _, done, _ = env.step(action)
        applied += 1
        if done:
            break
    return applied


def evaluate_policy(
    env: WorldsHardestGameEnv,
    q_table: DefaultDict[State, np.ndarray],
    episodes: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    wins = 0
    deaths = 0
    total_steps = 0

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = greedy_action(q_table[state], rng)
            state, _, done, info = env.step(action)

        if info["result"] == "win":
            wins += 1
        elif info["result"] == "death":
            deaths += 1
        total_steps += int(info["macro_steps"])

    return {
        "win_rate": wins / float(episodes),
        "death_rate": deaths / float(episodes),
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
        q_values = np.array([q_table[state] for state in states], dtype=np.float32)
    else:
        state_array = np.empty((0, 3), dtype=np.int16)
        q_values = np.empty((0, len(WorldsHardestGameEnv.ACTIONS)), dtype=np.float32)

    np.savez_compressed(
        model_path,
        states=state_array,
        q_values=q_values,
        metadata=json.dumps(metadata),
        expert_actions=np.array(expert_actions, dtype=np.int8),
    )


def train(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    env = WorldsHardestGameEnv(
        action_repeat=args.action_repeat,
        max_macro_steps=args.max_macro_steps,
        terminate_on_death=True,
    )

    if not args.quiet:
        print(
            f"loaded env: period={env.period}, action_repeat={env.action_repeat}, "
            f"max_macro_steps={env.max_macro_steps}"
        )

    plan: PlanResult = plan_level_actions(
        env,
        max_expand=args.planner_max_expand,
        progress_interval=100_000,
        verbose=not args.quiet,
    )
    if not plan.found:
        raise RuntimeError("Planner failed to find an expert trajectory for level 1.")

    if not args.quiet:
        print(f"expert trajectory found: macro_actions={len(plan.actions)} expanded={plan.expanded}")

    q_table: DefaultDict[State, np.ndarray] = defaultdict(
        lambda: np.zeros(len(WorldsHardestGameEnv.ACTIONS), dtype=np.float32)
    )

    warmstart_count = warmstart_q_table(env, q_table, plan.actions, args.warmstart_bonus)
    if not args.quiet:
        print(f"warm-started {warmstart_count} macro states from expert path")

    best_eval_win_rate = -1.0
    recent_results: List[str] = []
    train_start = time.time()

    for episode_idx in range(args.episodes):
        epsilon = epsilon_for_episode(
            episode_idx,
            args.episodes,
            args.epsilon_start,
            args.epsilon_end,
        )

        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, len(WorldsHardestGameEnv.ACTIONS)))
            else:
                action = greedy_action(q_table[state], rng)

            next_state, reward, done, info = env.step(action)
            total_reward += reward

            best_next = 0.0 if done else float(np.max(q_table[next_state]))
            target = reward + args.gamma * best_next
            q_table[state][action] += args.alpha * (target - q_table[state][action])

            state = next_state

        recent_results.append(str(info["result"]))
        if len(recent_results) > 200:
            recent_results = recent_results[-200:]

        if (episode_idx + 1) % args.eval_every == 0 or episode_idx == args.episodes - 1:
            metrics = evaluate_policy(env, q_table, args.eval_episodes, rng)
            best_eval_win_rate = max(best_eval_win_rate, metrics["win_rate"])

            if not args.quiet:
                recent_win_rate = recent_results.count("win") / float(len(recent_results))
                print(
                    f"ep={episode_idx + 1} "
                    f"epsilon={epsilon:.3f} "
                    f"train_recent_win={recent_win_rate:.3f} "
                    f"eval_win={metrics['win_rate']:.3f} "
                    f"eval_death={metrics['death_rate']:.3f} "
                    f"eval_avg_steps={metrics['avg_macro_steps']:.1f} "
                    f"q_states={len(q_table)} "
                    f"last_reward={total_reward:.3f}"
                )

    elapsed = time.time() - train_start
    final_metrics = evaluate_policy(env, q_table, max(50, args.eval_episodes), rng)

    metadata: Dict[str, object] = {
        "level": env.level_num,
        "period": env.period,
        "action_repeat": env.action_repeat,
        "max_macro_steps": env.max_macro_steps,
        "episodes": args.episodes,
        "seed": args.seed,
        "alpha": args.alpha,
        "gamma": args.gamma,
        "epsilon_start": args.epsilon_start,
        "epsilon_end": args.epsilon_end,
        "planner_expanded": plan.expanded,
        "expert_macro_actions": len(plan.actions),
        "warmstart_bonus": args.warmstart_bonus,
        "elapsed_sec": elapsed,
        "best_eval_win_rate": best_eval_win_rate,
        "final_eval_win_rate": final_metrics["win_rate"],
    }

    save_q_table(args.model_out, q_table, metadata, plan.actions)

    print(f"saved model: {args.model_out}")
    print(
        "final_eval "
        f"win_rate={final_metrics['win_rate']:.3f} "
        f"death_rate={final_metrics['death_rate']:.3f} "
        f"avg_macro_steps={final_metrics['avg_macro_steps']:.1f}"
    )


if __name__ == "__main__":
    train(parse_args())
