from __future__ import annotations

import heapq
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from whg_full_env import PlannerState, WorldsHardestGameFullEnv


@dataclass
class SegmentPlan:
    actions: List[int]
    expanded: int
    found: bool
    reason: str


@dataclass
class LevelPlan:
    actions: List[int]
    expanded: int
    segments: int
    found: bool
    result: str
    reason: str


@dataclass
class BeamNode:
    snapshot: Dict[str, Any]
    actions: List[int]
    expanded: int
    segments: int
    coins_collected: int


def _euclid(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _in_win_region(env: WorldsHardestGameFullEnv, x: int, y: int) -> bool:
    region = env.check_regions.get(env.win_check)
    if region is None:
        return False
    return region.x0 <= x <= region.x1 and region.y0 <= y <= region.y1


def _goal_heuristic(state: PlannerState, target: Tuple[float, float], action_repeat: int) -> float:
    x, y, _ = state
    manhattan = abs(float(x) - target[0]) + abs(float(y) - target[1])
    return manhattan / max(1.0, float(action_repeat))


def _plan_shortest_to_predicate(
    env: WorldsHardestGameFullEnv,
    start: PlannerState,
    predicate,
    allow_wait: bool = True,
    max_expand: int | None = None,
) -> SegmentPlan:
    actions = list(WorldsHardestGameFullEnv.ACTIONS)
    if not allow_wait and WorldsHardestGameFullEnv.ACTION_STAY in actions:
        actions.remove(WorldsHardestGameFullEnv.ACTION_STAY)

    parent: Dict[PlannerState, Optional[PlannerState]] = {start: None}
    parent_action: Dict[PlannerState, int] = {}
    q = deque([start])
    expanded = 0
    found_state: PlannerState | None = None

    while q:
        if max_expand is not None and expanded >= max_expand:
            break
        state = q.popleft()
        expanded += 1
        if predicate(state):
            found_state = state
            break

        for action in actions:
            next_state, dead, _ = env.macro_transition_from(state, action, repeat=env.action_repeat)
            if dead or next_state in parent:
                continue
            parent[next_state] = state
            parent_action[next_state] = action
            q.append(next_state)

    if found_state is None:
        reason = "max_expand" if (max_expand is not None and expanded >= max_expand) else "no_path"
        return SegmentPlan(actions=[], expanded=expanded, found=False, reason=reason)

    path_actions = _reconstruct_actions(parent, parent_action, found_state)
    return SegmentPlan(actions=path_actions, expanded=expanded, found=True, reason="ok")


def _reconstruct_actions(
    parent: Dict[PlannerState, Optional[PlannerState]],
    parent_action: Dict[PlannerState, int],
    goal_state: PlannerState,
) -> List[int]:
    actions: List[int] = []
    cur = goal_state
    while parent[cur] is not None:
        actions.append(parent_action[cur])
        cur = parent[cur]  # type: ignore[index]
    actions.reverse()
    return actions


def plan_segment_to_target(
    env: WorldsHardestGameFullEnv,
    target_xy: Tuple[float, float],
    target_radius: float,
    max_expand: int = 220_000,
    allow_wait: bool = True,
    action_repeat: int | None = None,
    verbose: bool = False,
) -> SegmentPlan:
    start = env.get_planner_state()
    repeat = env.action_repeat if action_repeat is None else int(action_repeat)
    actions = list(WorldsHardestGameFullEnv.ACTIONS)
    if not allow_wait and WorldsHardestGameFullEnv.ACTION_STAY in actions:
        actions.remove(WorldsHardestGameFullEnv.ACTION_STAY)

    open_heap: List[Tuple[float, int, PlannerState]] = []
    start_h = _goal_heuristic(start, target_xy, repeat)
    heapq.heappush(open_heap, (start_h, 0, start))

    parent: Dict[PlannerState, Optional[PlannerState]] = {start: None}
    parent_action: Dict[PlannerState, int] = {}
    distance: Dict[PlannerState, int] = {start: 0}

    expanded = 0
    found_state: PlannerState | None = None

    while open_heap and expanded < max_expand:
        f, g, state = heapq.heappop(open_heap)
        if g != distance[state]:
            continue

        expanded += 1
        if verbose and expanded % 25_000 == 0:
            print(f"planner segment expanded={expanded} open={len(open_heap)} f={f:.2f} g={g}")

        x, y, _ = state
        if _euclid((float(x), float(y)), target_xy) <= target_radius:
            found_state = state
            break

        for action in actions:
            next_state, dead, _ = env.macro_transition_from(state, action, repeat=repeat)
            if dead:
                continue

            next_g = g + 1
            if next_g >= distance.get(next_state, 10**9):
                continue

            distance[next_state] = next_g
            parent[next_state] = state
            parent_action[next_state] = action

            next_f = next_g + _goal_heuristic(next_state, target_xy, repeat)
            heapq.heappush(open_heap, (next_f, next_g, next_state))

    if found_state is None:
        return SegmentPlan(actions=[], expanded=expanded, found=False, reason="no_path")

    path_actions = _reconstruct_actions(parent, parent_action, found_state)
    return SegmentPlan(actions=path_actions, expanded=expanded, found=True, reason="ok")


def plan_segment_to_win_region(
    env: WorldsHardestGameFullEnv,
    max_expand: int = 220_000,
    allow_wait: bool = True,
    action_repeat: int | None = None,
    verbose: bool = False,
) -> SegmentPlan:
    start = env.get_planner_state()
    repeat = env.action_repeat if action_repeat is None else int(action_repeat)
    win_target = env.win_center()
    actions = list(WorldsHardestGameFullEnv.ACTIONS)
    if not allow_wait and WorldsHardestGameFullEnv.ACTION_STAY in actions:
        actions.remove(WorldsHardestGameFullEnv.ACTION_STAY)

    open_heap: List[Tuple[float, int, PlannerState]] = []
    start_h = _goal_heuristic(start, win_target, repeat)
    heapq.heappush(open_heap, (start_h, 0, start))

    parent: Dict[PlannerState, Optional[PlannerState]] = {start: None}
    parent_action: Dict[PlannerState, int] = {}
    distance: Dict[PlannerState, int] = {start: 0}

    expanded = 0
    found_state: PlannerState | None = None

    while open_heap and expanded < max_expand:
        f, g, state = heapq.heappop(open_heap)
        if g != distance[state]:
            continue

        expanded += 1
        if verbose and expanded % 25_000 == 0:
            print(f"planner win segment expanded={expanded} open={len(open_heap)} f={f:.2f} g={g}")

        x, y, _ = state
        if _in_win_region(env, x, y):
            found_state = state
            break

        for action in actions:
            next_state, dead, _ = env.macro_transition_from(state, action, repeat=repeat)
            if dead:
                continue

            next_g = g + 1
            if next_g >= distance.get(next_state, 10**9):
                continue

            distance[next_state] = next_g
            parent[next_state] = state
            parent_action[next_state] = action
            next_f = next_g + _goal_heuristic(next_state, win_target, repeat)
            heapq.heappush(open_heap, (next_f, next_g, next_state))

    if found_state is None:
        return SegmentPlan(actions=[], expanded=expanded, found=False, reason="no_path")

    path_actions = _reconstruct_actions(parent, parent_action, found_state)
    return SegmentPlan(actions=path_actions, expanded=expanded, found=True, reason="ok")


def _next_coin_target(env: WorldsHardestGameFullEnv) -> Tuple[float, float] | None:
    remaining = env.remaining_coin_positions()
    if remaining.shape[0] == 0:
        return None
    px, py, _ = env.get_planner_state()
    diff = remaining - np.array([float(px), float(py)], dtype=np.float32)[None, :]
    dist_sq = (diff**2).sum(axis=1)
    idx = int(np.argmin(dist_sq))
    return (float(remaining[idx, 0]), float(remaining[idx, 1]))


def _candidate_coin_targets(env: WorldsHardestGameFullEnv, limit: int = 3) -> List[Tuple[float, float]]:
    remaining = env.remaining_coin_positions()
    if remaining.shape[0] == 0:
        return []
    limit = max(1, int(limit))
    px, py, _ = env.get_planner_state()
    diff = remaining - np.array([float(px), float(py)], dtype=np.float32)[None, :]
    dist_sq = (diff**2).sum(axis=1)
    order = np.argsort(dist_sq)

    targets: List[Tuple[float, float]] = []
    for idx in order[:limit].tolist():
        targets.append((float(remaining[idx, 0]), float(remaining[idx, 1])))

    if remaining.shape[0] > limit:
        far_idx = int(order[-1])
        far = (float(remaining[far_idx, 0]), float(remaining[far_idx, 1]))
        if far not in targets:
            targets.append(far)
    return targets


def _expand_schedule(base_expand: int, retry_expand_cap: int) -> List[int]:
    base = max(1, int(base_expand))
    cap = max(base, int(retry_expand_cap))
    vals = {
        base,
        min(cap, base * 2),
        min(cap, base * 4),
        cap,
    }
    return sorted(vals)


def _expand_actions(actions: List[int], repeat: int) -> List[int]:
    r = max(1, int(repeat))
    if r == 1:
        return list(actions)
    out: List[int] = []
    for action in actions:
        out.extend([action] * r)
    return out


def _plan_static_actions_to_target(
    env: WorldsHardestGameFullEnv,
    target_xy: Tuple[float, float],
    target_radius: float,
    max_expand: int,
    allow_wait: bool,
) -> SegmentPlan:
    start = env.get_planner_state()
    sx, sy, _ = start
    actions = list(WorldsHardestGameFullEnv.ACTIONS)
    if not allow_wait and WorldsHardestGameFullEnv.ACTION_STAY in actions:
        actions.remove(WorldsHardestGameFullEnv.ACTION_STAY)

    start_xy = (int(sx), int(sy))
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_xy: None}
    parent_action: Dict[Tuple[int, int], int] = {}
    q = deque([start_xy])
    expanded = 0
    found_xy: Tuple[int, int] | None = None
    transition_cache: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

    while q and expanded < max_expand:
        xy = q.popleft()
        expanded += 1
        x, y = xy
        if _euclid((float(x), float(y)), target_xy) <= target_radius:
            found_xy = xy
            break

        for action in actions:
            key = (x, y, action)
            nxt_xy = transition_cache.get(key)
            if nxt_xy is None:
                nx, ny = env._apply_movement(float(x), float(y), action)
                nxt_xy = (int(round(nx)), int(round(ny)))
                transition_cache[key] = nxt_xy

            if nxt_xy in parent:
                continue
            parent[nxt_xy] = xy
            parent_action[nxt_xy] = action
            q.append(nxt_xy)

    if found_xy is None:
        reason = "max_expand" if expanded >= max_expand else "no_path"
        return SegmentPlan(actions=[], expanded=expanded, found=False, reason=reason)

    out_actions: List[int] = []
    cur = found_xy
    while parent[cur] is not None:
        out_actions.append(parent_action[cur])
        cur = parent[cur]  # type: ignore[index]
    out_actions.reverse()
    return SegmentPlan(actions=out_actions, expanded=expanded, found=True, reason="ok")


def _plan_static_actions_to_win_region(
    env: WorldsHardestGameFullEnv,
    max_expand: int,
    allow_wait: bool,
) -> SegmentPlan:
    start = env.get_planner_state()
    sx, sy, _ = start
    actions = list(WorldsHardestGameFullEnv.ACTIONS)
    if not allow_wait and WorldsHardestGameFullEnv.ACTION_STAY in actions:
        actions.remove(WorldsHardestGameFullEnv.ACTION_STAY)

    start_xy = (int(sx), int(sy))
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_xy: None}
    parent_action: Dict[Tuple[int, int], int] = {}
    q = deque([start_xy])
    expanded = 0
    found_xy: Tuple[int, int] | None = None
    transition_cache: Dict[Tuple[int, int, int], Tuple[int, int]] = {}

    while q and expanded < max_expand:
        xy = q.popleft()
        expanded += 1
        x, y = xy
        if _in_win_region(env, x, y):
            found_xy = xy
            break

        for action in actions:
            key = (x, y, action)
            nxt_xy = transition_cache.get(key)
            if nxt_xy is None:
                nx, ny = env._apply_movement(float(x), float(y), action)
                nxt_xy = (int(round(nx)), int(round(ny)))
                transition_cache[key] = nxt_xy

            if nxt_xy in parent:
                continue
            parent[nxt_xy] = xy
            parent_action[nxt_xy] = action
            q.append(nxt_xy)

    if found_xy is None:
        reason = "max_expand" if expanded >= max_expand else "no_path"
        return SegmentPlan(actions=[], expanded=expanded, found=False, reason=reason)

    out_actions: List[int] = []
    cur = found_xy
    while parent[cur] is not None:
        out_actions.append(parent_action[cur])
        cur = parent[cur]  # type: ignore[index]
    out_actions.reverse()
    return SegmentPlan(actions=out_actions, expanded=expanded, found=True, reason="ok")


def _time_align_actions_without_detour(
    env: WorldsHardestGameFullEnv,
    start: PlannerState,
    base_actions: List[int],
) -> SegmentPlan:
    if not base_actions:
        return SegmentPlan(actions=[], expanded=0, found=True, reason="ok")

    sx, sy, sphase = start
    nominal_positions: List[Tuple[int, int]] = [(int(sx), int(sy))]
    x, y = int(sx), int(sy)
    for action in base_actions:
        nx, ny = env._apply_movement(float(x), float(y), action)
        x, y = int(round(nx)), int(round(ny))
        nominal_positions.append((x, y))

    goal_idx = len(base_actions)
    start_node = (0, int(sphase))
    q = deque([start_node])
    parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start_node: None}
    parent_action: Dict[Tuple[int, int], int] = {}
    expanded = 0
    found_node: Tuple[int, int] | None = None
    max_nodes = min(200_000, max(1, (goal_idx + 1) * max(1, env.phase_mod)))

    while q and expanded < max_nodes:
        node = q.popleft()
        idx, phase = node
        expanded += 1
        if idx == goal_idx:
            found_node = node
            break

        x, y = nominal_positions[idx]
        planner_state = (x, y, phase)

        # Wait transition.
        wait_state, wait_dead, _ = env.macro_transition_from(planner_state, WorldsHardestGameFullEnv.ACTION_STAY, repeat=1)
        if not wait_dead:
            wx, wy, wphase = wait_state
            if (wx, wy) == (x, y):
                nxt = (idx, int(wphase))
                if nxt not in parent:
                    parent[nxt] = node
                    parent_action[nxt] = WorldsHardestGameFullEnv.ACTION_STAY
                    q.append(nxt)

        # Advance transition along nominal static path.
        action = int(base_actions[idx])
        adv_state, adv_dead, _ = env.macro_transition_from(planner_state, action, repeat=1)
        if not adv_dead:
            ax, ay, aphase = adv_state
            if (ax, ay) == nominal_positions[idx + 1]:
                nxt = (idx + 1, int(aphase))
                if nxt not in parent:
                    parent[nxt] = node
                    parent_action[nxt] = action
                    q.append(nxt)

    if found_node is None:
        return SegmentPlan(actions=[], expanded=expanded, found=False, reason="no_path")

    out_actions: List[int] = []
    cur = found_node
    while parent[cur] is not None:
        out_actions.append(parent_action[cur])
        cur = parent[cur]  # type: ignore[index]
    out_actions.reverse()
    return SegmentPlan(actions=out_actions, expanded=expanded, found=True, reason="ok")


def _plan_coin_segment_with_retries(
    env: WorldsHardestGameFullEnv,
    max_expand_per_segment: int,
    retry_expand_cap: int,
    allow_wait: bool,
    verbose: bool,
    coin_target_candidates: int,
) -> SegmentPlan:
    schedule = _expand_schedule(max_expand_per_segment, retry_expand_cap)
    total_expanded = 0
    last_reason = "no_path"
    # Multi-resolution planning often finds valid timed paths on hard levels
    # much faster than strict repeat=1 search alone.
    planning_repeats = (3, 2, 1)

    for budget in schedule:
        targets = _candidate_coin_targets(env, limit=coin_target_candidates)
        if not targets:
            break

        best_static_actions: List[int] | None = None
        for target in targets:
            static_segment = _plan_static_actions_to_target(
                env=env,
                target_xy=target,
                target_radius=env.coin_collect_radius,
                max_expand=budget,
                allow_wait=allow_wait,
            )
            total_expanded += static_segment.expanded
            if not static_segment.found:
                last_reason = f"static_{static_segment.reason}"
                continue

            timed = _time_align_actions_without_detour(
                env=env,
                start=env.get_planner_state(),
                base_actions=static_segment.actions,
            )
            total_expanded += timed.expanded
            if timed.found:
                if best_static_actions is None or len(timed.actions) < len(best_static_actions):
                    best_static_actions = timed.actions
            else:
                last_reason = f"static_time_{timed.reason}"

        if best_static_actions is not None:
            return SegmentPlan(
                actions=best_static_actions,
                expanded=total_expanded,
                found=True,
                reason=f"static_ok@{budget}",
            )

        for repeat in planning_repeats:
            best_actions: List[int] | None = None
            for target in targets:
                segment = plan_segment_to_target(
                    env=env,
                    target_xy=target,
                    target_radius=env.coin_collect_radius,
                    max_expand=budget,
                    allow_wait=allow_wait,
                    action_repeat=repeat,
                    verbose=verbose,
                )
                total_expanded += segment.expanded
                if segment.found:
                    expanded_actions = _expand_actions(segment.actions, repeat)
                    if best_actions is None or len(expanded_actions) < len(best_actions):
                        best_actions = expanded_actions
                else:
                    last_reason = segment.reason

            if best_actions is not None:
                return SegmentPlan(
                    actions=best_actions,
                    expanded=total_expanded,
                    found=True,
                    reason=f"ok@{budget}x{repeat}",
                )

    return SegmentPlan(
        actions=[],
        expanded=total_expanded,
        found=False,
        reason=f"{last_reason}@{schedule[-1]}",
    )


def _plan_coin_segment_candidates(
    env: WorldsHardestGameFullEnv,
    max_expand_per_segment: int,
    retry_expand_cap: int,
    allow_wait: bool,
    verbose: bool,
    coin_target_candidates: int,
    max_candidates: int,
) -> Tuple[List[SegmentPlan], int]:
    schedule = _expand_schedule(max_expand_per_segment, retry_expand_cap)
    planning_repeats = (3, 2, 1)
    total_expanded = 0
    scored: List[Tuple[int, SegmentPlan]] = []
    seen: set[Tuple[int, Tuple[int, ...]]] = set()

    for budget in schedule:
        targets = _candidate_coin_targets(env, limit=coin_target_candidates)
        if not targets:
            break

        for target in targets:
            static_segment = _plan_static_actions_to_target(
                env=env,
                target_xy=target,
                target_radius=env.coin_collect_radius,
                max_expand=budget,
                allow_wait=allow_wait,
            )
            total_expanded += static_segment.expanded
            if static_segment.found:
                timed = _time_align_actions_without_detour(
                    env=env,
                    start=env.get_planner_state(),
                    base_actions=static_segment.actions,
                )
                total_expanded += timed.expanded
                if timed.found and timed.actions:
                    key = (len(timed.actions), tuple(timed.actions[:24]))
                    if key not in seen:
                        seen.add(key)
                        scored.append(
                            (
                                len(timed.actions),
                                SegmentPlan(
                                    actions=timed.actions,
                                    expanded=static_segment.expanded + timed.expanded,
                                    found=True,
                                    reason=f"static_ok@{budget}",
                                ),
                            )
                        )

            for repeat in planning_repeats:
                segment = plan_segment_to_target(
                    env=env,
                    target_xy=target,
                    target_radius=env.coin_collect_radius,
                    max_expand=budget,
                    allow_wait=allow_wait,
                    action_repeat=repeat,
                    verbose=verbose,
                )
                total_expanded += segment.expanded
                if not segment.found:
                    continue
                expanded_actions = _expand_actions(segment.actions, repeat)
                if not expanded_actions:
                    continue
                key = (len(expanded_actions), tuple(expanded_actions[:24]))
                if key in seen:
                    continue
                seen.add(key)
                scored.append(
                    (
                        len(expanded_actions),
                        SegmentPlan(
                            actions=expanded_actions,
                            expanded=segment.expanded,
                            found=True,
                            reason=f"ok@{budget}x{repeat}",
                        ),
                    )
                )

    scored.sort(key=lambda v: int(v[0]))
    out = [seg for _, seg in scored[: max(1, int(max_candidates))]]
    return out, total_expanded


def _plan_win_segment_with_retries(
    env: WorldsHardestGameFullEnv,
    max_expand_per_segment: int,
    retry_expand_cap: int,
    allow_wait: bool,
    verbose: bool,
) -> SegmentPlan:
    schedule = _expand_schedule(max_expand_per_segment, retry_expand_cap)
    total_expanded = 0
    last_reason = "no_path"
    # Multi-resolution planning often finds valid timed paths on hard levels
    # much faster than strict repeat=1 search alone.
    planning_repeats = (3, 2, 1)

    for budget in schedule:
        static_segment = _plan_static_actions_to_win_region(
            env=env,
            max_expand=budget,
            allow_wait=allow_wait,
        )
        total_expanded += static_segment.expanded
        if static_segment.found:
            timed = _time_align_actions_without_detour(
                env=env,
                start=env.get_planner_state(),
                base_actions=static_segment.actions,
            )
            total_expanded += timed.expanded
            if timed.found:
                return SegmentPlan(
                    actions=timed.actions,
                    expanded=total_expanded,
                    found=True,
                    reason=f"static_ok@{budget}",
                )
            last_reason = f"static_time_{timed.reason}"
        else:
            last_reason = f"static_{static_segment.reason}"

        for repeat in planning_repeats:
            win_segment = plan_segment_to_win_region(
                env=env,
                max_expand=budget,
                allow_wait=allow_wait,
                action_repeat=repeat,
                verbose=verbose,
            )
            total_expanded += win_segment.expanded
            if win_segment.found:
                return SegmentPlan(
                    actions=_expand_actions(win_segment.actions, repeat),
                    expanded=total_expanded,
                    found=True,
                    reason=f"ok@{budget}x{repeat}",
                )

        fallback = _plan_shortest_to_predicate(
            env=env,
            start=env.get_planner_state(),
            predicate=lambda s: _in_win_region(env, s[0], s[1]),
            allow_wait=allow_wait,
            max_expand=budget,
        )
        total_expanded += fallback.expanded
        if fallback.found:
            return SegmentPlan(
                actions=fallback.actions,
                expanded=total_expanded,
                found=True,
                reason=f"fallback_ok@{budget}",
            )
        last_reason = f"{win_segment.reason}|fallback:{fallback.reason}"

    return SegmentPlan(
        actions=[],
        expanded=total_expanded,
        found=False,
        reason=f"{last_reason}@{schedule[-1]}",
    )


def _capture_env_state(env: WorldsHardestGameFullEnv) -> Dict[str, Any]:
    return {
        "x": float(env.x),
        "y": float(env.y),
        "phase": int(env.phase),
        "current_checkpoint_idx": int(env.current_checkpoint_idx),
        "current_checkpoint_name": str(env.current_checkpoint_name),
        "current_spawn": (float(env.current_spawn[0]), float(env.current_spawn[1])),
        "coin_collected": env.coin_collected.copy(),
        "prev_objective_distance": float(env.prev_objective_distance),
        "done": bool(env.done),
        "last_result": str(env.last_result),
        "macro_steps": int(env.macro_steps),
        "total_micro_steps": int(env.total_micro_steps),
        "deaths": int(env.deaths),
    }


def _restore_env_state(env: WorldsHardestGameFullEnv, snap: Dict[str, Any]) -> None:
    env.x = float(snap["x"])
    env.y = float(snap["y"])
    env.phase = int(snap["phase"])
    env.current_checkpoint_idx = int(snap["current_checkpoint_idx"])
    env.current_checkpoint_name = str(snap["current_checkpoint_name"])
    sx, sy = snap["current_spawn"]
    env.current_spawn = (float(sx), float(sy))
    env.coin_collected = np.asarray(snap["coin_collected"], dtype=bool).copy()
    env.prev_objective_distance = float(snap["prev_objective_distance"])
    env.done = bool(snap["done"])
    env.last_result = str(snap["last_result"])
    env.macro_steps = int(snap["macro_steps"])
    env.total_micro_steps = int(snap["total_micro_steps"])
    env.deaths = int(snap["deaths"])


def _state_visit_key(env: WorldsHardestGameFullEnv) -> Tuple[int, int, int, bytes]:
    x, y, phase = env.get_planner_state()
    return (int(x), int(y), int(phase), bytes(env.coin_collected.tobytes()))


def plan_level_coin_beam_fallback(
    env: WorldsHardestGameFullEnv,
    max_expand_per_segment: int,
    retry_expand_cap: int,
    max_segments: int,
    allow_wait: bool,
    verbose: bool,
    coin_target_candidates: int,
    beam_width: int = 6,
    beam_branch: int = 12,
    beam_max_nodes: int = 600,
) -> LevelPlan:
    env.reset()
    root = BeamNode(
        snapshot=_capture_env_state(env),
        actions=[],
        expanded=0,
        segments=0,
        coins_collected=int(env.coins_collected_count),
    )
    frontier: List[Tuple[Tuple[int, int, int], int, BeamNode]] = [((0, 0, 0), 0, root)]
    push_id = 0
    best = root
    visited_best_len: Dict[Tuple[int, int, int, bytes], int] = {_state_visit_key(env): 0}
    total_expanded = 0
    explored_nodes = 0

    while frontier and explored_nodes < max(1, int(beam_max_nodes)):
        _, _, node = heapq.heappop(frontier)
        if node.coins_collected > best.coins_collected:
            best = node

        if node.segments >= max_segments:
            continue

        _restore_env_state(env, node.snapshot)
        if env.coins_remaining_count <= 0:
            to_win = _plan_win_segment_with_retries(
                env=env,
                max_expand_per_segment=max_expand_per_segment,
                retry_expand_cap=retry_expand_cap,
                allow_wait=allow_wait,
                verbose=verbose,
            )
            total_expanded += to_win.expanded
            if to_win.found:
                out_actions = list(node.actions)
                for action in to_win.actions:
                    _, _, done, info = env.step(action, capture_frames=False)
                    out_actions.append(action)
                    if done:
                        return LevelPlan(
                            actions=out_actions,
                            expanded=total_expanded,
                            segments=node.segments + 1,
                            found=(info["result"] == "win"),
                            result=str(info["result"]),
                            reason="beam_ok" if info["result"] == "win" else "beam_terminated_before_win",
                        )
            continue

        candidates, expanded_here = _plan_coin_segment_candidates(
            env=env,
            max_expand_per_segment=max_expand_per_segment,
            retry_expand_cap=retry_expand_cap,
            allow_wait=allow_wait,
            verbose=verbose,
            coin_target_candidates=max(int(coin_target_candidates), 48),
            max_candidates=max(1, int(beam_branch)),
        )
        total_expanded += expanded_here
        explored_nodes += 1
        if not candidates:
            continue

        for seg in candidates:
            _restore_env_state(env, node.snapshot)
            before = int(env.coins_collected_count)
            out_actions = list(node.actions)
            died = False

            for action in seg.actions:
                _, _, done, info = env.step(action, capture_frames=False)
                out_actions.append(action)
                if done:
                    if info["result"] == "win":
                        return LevelPlan(
                            actions=out_actions,
                            expanded=total_expanded,
                            segments=node.segments + 1,
                            found=True,
                            result="win",
                            reason="beam_ok",
                        )
                    died = True
                    break
                if int(env.coins_collected_count) > before:
                    break

            if died:
                continue
            if int(env.coins_collected_count) <= before:
                continue

            key = _state_visit_key(env)
            best_len = visited_best_len.get(key)
            if best_len is not None and len(out_actions) >= best_len:
                continue
            visited_best_len[key] = len(out_actions)

            coins = int(env.coins_collected_count)
            child = BeamNode(
                snapshot=_capture_env_state(env),
                actions=out_actions,
                expanded=node.expanded + seg.expanded,
                segments=node.segments + 1,
                coins_collected=coins,
            )
            push_id += 1
            priority = (-coins, len(out_actions), child.segments)
            heapq.heappush(frontier, (priority, push_id, child))

    return LevelPlan(
        actions=list(best.actions),
        expanded=total_expanded,
        segments=int(best.segments),
        found=False,
        result=env.last_result,
        reason=f"beam_failed:coins={best.coins_collected}/{env.coins_required}",
    )


def _coin_mask_at_position(
    x: int,
    y: int,
    coin_positions: np.ndarray,
    coin_collect_radius_sq: float,
) -> int:
    if coin_positions.shape[0] == 0:
        return 0
    diff = coin_positions - np.array([float(x), float(y)], dtype=np.float32)[None, :]
    dist_sq = (diff**2).sum(axis=1)
    mask = 0
    for i, d in enumerate(dist_sq.tolist()):
        if d <= coin_collect_radius_sq:
            mask |= 1 << i
    return mask


def _heuristic_small_coin(
    x: int,
    y: int,
    mask: int,
    full_mask: int,
    coin_positions: np.ndarray,
    win_center: Tuple[float, float],
    action_repeat: int,
) -> float:
    if mask == full_mask:
        return (abs(float(x) - win_center[0]) + abs(float(y) - win_center[1])) / max(1.0, float(action_repeat))

    remaining: List[Tuple[float, float]] = []
    for i in range(coin_positions.shape[0]):
        if (mask >> i) & 1:
            continue
        remaining.append((float(coin_positions[i, 0]), float(coin_positions[i, 1])))
    if not remaining:
        return 0.0
    nearest = min(abs(float(x) - cx) + abs(float(y) - cy) for cx, cy in remaining)
    return nearest / max(1.0, float(action_repeat))


def plan_level_exhaustive_small_coins(
    env: WorldsHardestGameFullEnv,
    max_expand: int,
    allow_wait: bool = True,
    verbose: bool = False,
) -> LevelPlan:
    env.reset()
    n_coins = int(env.coin_positions.shape[0])
    full_mask = (1 << n_coins) - 1 if n_coins > 0 else 0
    coin_collect_radius_sq = float(env.coin_collect_radius * env.coin_collect_radius)
    win_center = env.win_center()

    start_x, start_y, start_phase = env.get_planner_state()
    start_mask = _coin_mask_at_position(start_x, start_y, env.coin_positions, coin_collect_radius_sq)
    start_state = (start_x, start_y, start_phase, start_mask)

    actions = list(WorldsHardestGameFullEnv.ACTIONS)
    if not allow_wait and WorldsHardestGameFullEnv.ACTION_STAY in actions:
        actions.remove(WorldsHardestGameFullEnv.ACTION_STAY)

    open_heap: List[Tuple[float, int, Tuple[int, int, int, int]]] = []
    start_h = _heuristic_small_coin(
        start_x,
        start_y,
        start_mask,
        full_mask,
        env.coin_positions,
        win_center,
        env.action_repeat,
    )
    heapq.heappush(open_heap, (start_h, 0, start_state))

    parent: Dict[Tuple[int, int, int, int], Optional[Tuple[int, int, int, int]]] = {start_state: None}
    parent_action: Dict[Tuple[int, int, int, int], int] = {}
    distance: Dict[Tuple[int, int, int, int], int] = {start_state: 0}

    expanded = 0
    found_state: Tuple[int, int, int, int] | None = None

    while open_heap and expanded < max_expand:
        f, g, state = heapq.heappop(open_heap)
        if g != distance[state]:
            continue

        expanded += 1
        if verbose and expanded % 50_000 == 0:
            print(f"planner exhaustive expanded={expanded} open={len(open_heap)} f={f:.2f} g={g}")

        x, y, phase, mask = state
        if mask == full_mask and _in_win_region(env, x, y):
            found_state = state
            break

        planner_state: PlannerState = (x, y, phase)
        for action in actions:
            next_xyz, dead, _ = env.macro_transition_from(planner_state, action, repeat=env.action_repeat)
            if dead:
                continue

            nx, ny, nphase = next_xyz
            collect_mask = _coin_mask_at_position(nx, ny, env.coin_positions, coin_collect_radius_sq)
            nmask = mask | collect_mask
            next_state = (nx, ny, nphase, nmask)

            next_g = g + 1
            if next_g >= distance.get(next_state, 10**9):
                continue

            distance[next_state] = next_g
            parent[next_state] = state
            parent_action[next_state] = action
            next_h = _heuristic_small_coin(nx, ny, nmask, full_mask, env.coin_positions, win_center, env.action_repeat)
            heapq.heappush(open_heap, (next_g + next_h, next_g, next_state))

    if found_state is None:
        return LevelPlan(
            actions=[],
            expanded=expanded,
            segments=1,
            found=False,
            result=env.last_result,
            reason="exhaustive_no_path",
        )

    actions_out: List[int] = []
    cur = found_state
    while parent[cur] is not None:
        actions_out.append(parent_action[cur])
        cur = parent[cur]  # type: ignore[index]
    actions_out.reverse()

    env.reset()
    result = env.last_result
    found = False
    for action in actions_out:
        _, _, done, info = env.step(action, capture_frames=False)
        if done:
            result = str(info["result"])
            found = result == "win"
            break
    if not found:
        return LevelPlan(
            actions=actions_out,
            expanded=expanded,
            segments=1,
            found=False,
            result=result,
            reason="exhaustive_path_not_win",
        )

    return LevelPlan(
        actions=actions_out,
        expanded=expanded,
        segments=1,
        found=True,
        result="win",
        reason="ok",
    )


def plan_level_no_enemy_static(
    env: WorldsHardestGameFullEnv,
    max_segments: int = 260,
    allow_wait: bool = True,
) -> LevelPlan:
    env.reset()
    all_actions: List[int] = []
    total_expanded = 0
    segments = 0

    while env.coins_remaining_count > 0 and segments < max_segments:
        target = _next_coin_target(env)
        if target is None:
            break

        def coin_goal(state: PlannerState) -> bool:
            x, y, _ = state
            return _euclid((float(x), float(y)), target) <= env.coin_collect_radius

        segment = _plan_shortest_to_predicate(
            env=env,
            start=env.get_planner_state(),
            predicate=coin_goal,
            allow_wait=allow_wait,
        )
        total_expanded += segment.expanded
        if not segment.found:
            return LevelPlan(
                actions=all_actions,
                expanded=total_expanded,
                segments=segments,
                found=False,
                result=env.last_result,
                reason=f"coin_segment_failed:{segment.reason}",
            )

        before = env.coins_collected_count
        for action in segment.actions:
            _, _, done, info = env.step(action, capture_frames=False)
            all_actions.append(action)
            if done:
                return LevelPlan(
                    actions=all_actions,
                    expanded=total_expanded,
                    segments=segments + 1,
                    found=(info["result"] == "win"),
                    result=str(info["result"]),
                    reason="terminated_during_coin_segment",
                )
            if env.coins_collected_count > before:
                break

        if env.coins_collected_count == before:
            return LevelPlan(
                actions=all_actions,
                expanded=total_expanded,
                segments=segments + 1,
                found=False,
                result=env.last_result,
                reason="coin_not_collected_after_segment",
            )
        segments += 1

    if env.coins_remaining_count > 0:
        return LevelPlan(
            actions=all_actions,
            expanded=total_expanded,
            segments=segments,
            found=False,
            result=env.last_result,
            reason="segment_limit_reached",
        )

    def win_goal(state: PlannerState) -> bool:
        x, y, _ = state
        return _in_win_region(env, x, y)

    to_win = _plan_shortest_to_predicate(
        env=env,
        start=env.get_planner_state(),
        predicate=win_goal,
        allow_wait=allow_wait,
    )
    total_expanded += to_win.expanded
    if not to_win.found:
        return LevelPlan(
            actions=all_actions,
            expanded=total_expanded,
            segments=segments,
            found=False,
            result=env.last_result,
            reason=f"win_segment_failed:{to_win.reason}",
        )

    for action in to_win.actions:
        _, _, done, info = env.step(action, capture_frames=False)
        all_actions.append(action)
        if done:
            return LevelPlan(
                actions=all_actions,
                expanded=total_expanded,
                segments=segments + 1,
                found=(info["result"] == "win"),
                result=str(info["result"]),
                reason="ok" if info["result"] == "win" else "terminated_before_win",
            )

    return LevelPlan(
        actions=all_actions,
        expanded=total_expanded,
        segments=segments + 1,
        found=False,
        result=env.last_result,
        reason="win_not_reached_after_segment",
    )


def plan_level_greedy(
    env: WorldsHardestGameFullEnv,
    max_expand_per_segment: int = 220_000,
    max_segments: int = 260,
    allow_wait: bool = True,
    verbose: bool = False,
    exact_small_coin_limit: int = 8,
    coin_target_candidates: int = 24,
    retry_expand_cap: int = 2_200_000,
) -> LevelPlan:
    env.reset()

    if env.ignore_enemies:
        static = plan_level_no_enemy_static(
            env=env,
            max_segments=max_segments,
            allow_wait=allow_wait,
        )
        if static.found:
            return static

    if 1 < env.coins_required <= min(2, exact_small_coin_limit):
        exhaustive_total_expanded = 0
        exhaustive_cap = min(int(retry_expand_cap), max(1, int(max_expand_per_segment) * 2))
        for budget in _expand_schedule(max_expand_per_segment, exhaustive_cap)[:2]:
            exhaustive = plan_level_exhaustive_small_coins(
                env=env,
                max_expand=budget,
                allow_wait=allow_wait,
                verbose=verbose,
            )
            exhaustive_total_expanded += exhaustive.expanded
            if exhaustive.found:
                return LevelPlan(
                    actions=exhaustive.actions,
                    expanded=exhaustive_total_expanded,
                    segments=exhaustive.segments,
                    found=True,
                    result=exhaustive.result,
                    reason=exhaustive.reason,
                )

    all_actions: List[int] = []
    total_expanded = 0
    segments = 0

    while env.coins_remaining_count > 0 and segments < max_segments:
        if env.coins_remaining_count <= 0:
            break

        segment = _plan_coin_segment_with_retries(
            env=env,
            max_expand_per_segment=max_expand_per_segment,
            retry_expand_cap=retry_expand_cap,
            allow_wait=allow_wait,
            verbose=verbose,
            coin_target_candidates=coin_target_candidates,
        )
        total_expanded += segment.expanded
        if not segment.found:
            if (not env.ignore_enemies) and env.coins_required >= 20:
                beam = plan_level_coin_beam_fallback(
                    env=env,
                    max_expand_per_segment=max_expand_per_segment,
                    retry_expand_cap=retry_expand_cap,
                    max_segments=max_segments,
                    allow_wait=allow_wait,
                    verbose=verbose,
                    coin_target_candidates=max(coin_target_candidates, 24),
                )
                if beam.found:
                    return beam
            return LevelPlan(
                actions=all_actions,
                expanded=total_expanded,
                segments=segments,
                found=False,
                result=env.last_result,
                reason=f"coin_segment_failed:{segment.reason}",
            )

        before = env.coins_collected_count
        for action in segment.actions:
            _, _, done, info = env.step(action, capture_frames=False)
            all_actions.append(action)
            if done:
                return LevelPlan(
                    actions=all_actions,
                    expanded=total_expanded,
                    segments=segments + 1,
                    found=(info["result"] == "win"),
                    result=str(info["result"]),
                    reason="terminated_during_coin_segment",
                )
            if env.coins_collected_count > before:
                break

        if env.coins_collected_count == before:
            return LevelPlan(
                actions=all_actions,
                expanded=total_expanded,
                segments=segments + 1,
                found=False,
                result=env.last_result,
                reason="coin_not_collected_after_segment",
            )

        segments += 1

    if env.coins_remaining_count > 0:
        return LevelPlan(
            actions=all_actions,
            expanded=total_expanded,
            segments=segments,
            found=False,
            result=env.last_result,
            reason="segment_limit_reached",
        )

    to_win = _plan_win_segment_with_retries(
        env=env,
        max_expand_per_segment=max_expand_per_segment,
        retry_expand_cap=retry_expand_cap,
        allow_wait=allow_wait,
        verbose=verbose,
    )
    total_expanded += to_win.expanded
    if not to_win.found:
        return LevelPlan(
            actions=all_actions,
            expanded=total_expanded,
            segments=segments,
            found=False,
            result=env.last_result,
            reason=f"win_segment_failed:{to_win.reason}",
        )

    for action in to_win.actions:
        _, _, done, info = env.step(action, capture_frames=False)
        all_actions.append(action)
        if done:
            return LevelPlan(
                actions=all_actions,
                expanded=total_expanded,
                segments=segments + 1,
                found=(info["result"] == "win"),
                result=str(info["result"]),
                reason="ok" if info["result"] == "win" else "terminated_before_win",
            )

    return LevelPlan(
        actions=all_actions,
        expanded=total_expanded,
        segments=segments + 1,
        found=False,
        result=env.last_result,
        reason="win_not_reached_after_segment",
    )
