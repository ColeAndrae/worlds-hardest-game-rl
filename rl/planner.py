from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from whg_env import WorldsHardestGameEnv


@dataclass
class PlanResult:
    actions: List[int]
    expanded: int
    found: bool


def plan_level_actions(
    env: WorldsHardestGameEnv,
    max_expand: int = 1_200_000,
    progress_interval: int = 100_000,
    verbose: bool = True,
) -> PlanResult:
    """A* planner over macro actions. Used for expert warm-start trajectories."""
    start = env.reset()
    open_heap: List[Tuple[int, int, Tuple[int, int, int]]] = []
    heapq.heappush(open_heap, (env.heuristic_to_goal(start[0], start[1]), 0, start))

    parent: Dict[Tuple[int, int, int], Optional[Tuple[int, int, int]]] = {start: None}
    parent_action: Dict[Tuple[int, int, int], int] = {}
    distance: Dict[Tuple[int, int, int], int] = {start: 0}

    expanded = 0
    found_state: Optional[Tuple[int, int, int]] = None

    while open_heap and expanded < max_expand:
        f, g, state = heapq.heappop(open_heap)
        if g != distance[state]:
            continue

        x, y, _ = state
        expanded += 1

        if verbose and expanded % progress_interval == 0:
            print(f"planner expanded={expanded} open={len(open_heap)}")

        if env.is_goal_position(x, y):
            found_state = state
            break

        for action in env.ACTIONS:
            next_state, dead, win, _ = env.macro_transition(state, action)
            if dead:
                continue

            next_g = g + 1
            if next_g < distance.get(next_state, 10**9):
                distance[next_state] = next_g
                parent[next_state] = state
                parent_action[next_state] = action

                h = env.heuristic_to_goal(next_state[0], next_state[1])
                heapq.heappush(open_heap, (next_g + h, next_g, next_state))

                if win:
                    found_state = next_state
                    open_heap = []
                    break

    if found_state is None:
        return PlanResult(actions=[], expanded=expanded, found=False)

    actions: List[int] = []
    cur = found_state
    while parent[cur] is not None:
        actions.append(parent_action[cur])
        cur = parent[cur]  # type: ignore[index]
    actions.reverse()

    return PlanResult(actions=actions, expanded=expanded, found=True)


def actions_to_micro(actions: List[int], repeat: int) -> List[int]:
    out: List[int] = []
    for action in actions:
        out.extend([action] * repeat)
    return out
