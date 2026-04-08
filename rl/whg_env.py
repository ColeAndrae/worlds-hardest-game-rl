from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class DotSpec:
    x: int
    y: int
    pos1: Tuple[int, int]
    pos2: Tuple[int, int]
    speed: float
    move_to_pos1: bool
    vertical: bool


@dataclass(frozen=True)
class DotState:
    x: float
    y: float
    move_to_pos1: bool


class WorldsHardestGameEnv:
    """Deterministic level-1 simulator compatible with tabular RL."""

    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTIONS: Tuple[int, ...] = (ACTION_STAY, ACTION_UP, ACTION_DOWN, ACTION_LEFT, ACTION_RIGHT)
    ACTION_NAMES: Dict[int, str] = {
        ACTION_STAY: "stay",
        ACTION_UP: "up",
        ACTION_DOWN: "down",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right",
    }

    BOARD_WIDTH = 800
    BOARD_HEIGHT = 600
    TILE_SIZE = 40

    def __init__(
        self,
        repo_root: str | Path | None = None,
        level_num: int = 1,
        action_repeat: int = 4,
        max_macro_steps: int = 260,
        step_penalty: float = -0.002,
        progress_reward_scale: float = 0.003,
        death_penalty: float = -1.0,
        goal_reward: float = 2.0,
        terminate_on_death: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[1]
        self.level_num = level_num
        self.action_repeat = action_repeat
        self.max_macro_steps = max_macro_steps

        self.step_penalty = step_penalty
        self.progress_reward_scale = progress_reward_scale
        self.death_penalty = death_penalty
        self.goal_reward = goal_reward
        self.terminate_on_death = terminate_on_death

        self.maps_dir = self.repo_root / "src" / "net" / "thedanpage" / "worldshardestgame" / "resources" / "maps"
        self.tile_types, self.dot_specs, self.spawn, self.goal_tiles = self._load_level(level_num)

        self.tile_width = len(self.tile_types[0])
        self.tile_height = len(self.tile_types)
        self.goal_bounds = [(gx * self.TILE_SIZE, gy * self.TILE_SIZE, 39, 39) for gx, gy in self.goal_tiles]
        self.goal_centers = [
            (gx * self.TILE_SIZE + self.TILE_SIZE // 2, gy * self.TILE_SIZE + self.TILE_SIZE // 2)
            for gx, gy in self.goal_tiles
        ]

        self.phase_dot_states = self._precompute_dot_cycle()
        self.period = len(self.phase_dot_states)
        self.dot_bounds_by_phase = self._precompute_dot_bounds()

        self.reset()

    @staticmethod
    def _parse_properties(text: str) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
        return out

    def _load_level(self, level_num: int) -> Tuple[List[List[int]], List[DotSpec], Tuple[int, int], List[Tuple[int, int]]]:
        map_path = self.maps_dir / f"level_{level_num}.txt"
        prop_path = self.maps_dir / f"level_{level_num}.properties"

        map_lines = map_path.read_text(encoding="utf-8").splitlines()
        prop = self._parse_properties(prop_path.read_text(encoding="utf-8"))

        tile_lines = map_lines[:15]
        tile_types: List[List[int]] = [[int(ch) for ch in line.strip()] for line in tile_lines]

        spawn_x, spawn_y = [int(v) for v in prop["spawn_point"].split(",")]
        spawn = (spawn_x * self.TILE_SIZE + self.TILE_SIZE // 2, spawn_y * self.TILE_SIZE + self.TILE_SIZE // 2)

        goal_tiles: List[Tuple[int, int]] = []
        for y, row in enumerate(tile_types):
            for x, tile in enumerate(row):
                if tile == 3:
                    goal_tiles.append((x, y))

        dot_specs: List[DotSpec] = []
        for line in map_lines[15:]:
            stripped = line.strip()
            if not stripped or stripped.startswith("Format") or stripped.startswith("IMPORTANT"):
                continue

            fields = [part.strip() for part in stripped.split("-")]
            x = int(fields[0])
            y = int(fields[1])
            pos1 = tuple(int(v) for v in fields[2].split(","))
            pos2 = tuple(int(v) for v in fields[3].split(","))
            speed = float(fields[4])
            move_to_pos1 = fields[5].lower() == "true"
            vertical = fields[6].lower() == "true"
            dot_specs.append(DotSpec(x, y, pos1, pos2, speed, move_to_pos1, vertical))

        return tile_types, dot_specs, spawn, goal_tiles

    def _initial_dot_states(self) -> Tuple[DotState, ...]:
        return tuple(
            DotState(
                x=float(spec.x * self.TILE_SIZE),
                y=float(spec.y * self.TILE_SIZE),
                move_to_pos1=spec.move_to_pos1,
            )
            for spec in self.dot_specs
        )

    def _step_dots(self, current: Sequence[DotState]) -> Tuple[DotState, ...]:
        out: List[DotState] = []
        for state, spec in zip(current, self.dot_specs):
            x = state.x
            y = state.y
            move_to_pos1 = state.move_to_pos1

            snap_x = int(x / self.TILE_SIZE)
            snap_y = int(y / self.TILE_SIZE)

            if move_to_pos1:
                if not spec.vertical:
                    x -= spec.speed
                else:
                    y -= spec.speed
                if snap_x < spec.pos1[0] or snap_y < spec.pos1[1]:
                    move_to_pos1 = False
            else:
                if not spec.vertical:
                    x += spec.speed
                else:
                    y += spec.speed
                if snap_x > spec.pos2[0] or snap_y > spec.pos2[1]:
                    move_to_pos1 = True

            out.append(DotState(x=x, y=y, move_to_pos1=move_to_pos1))
        return tuple(out)

    @staticmethod
    def _dot_cycle_key(states: Sequence[DotState]) -> Tuple[Tuple[float, float, bool], ...]:
        return tuple((round(dot.x, 6), round(dot.y, 6), dot.move_to_pos1) for dot in states)

    def _precompute_dot_cycle(self) -> List[Tuple[DotState, ...]]:
        states = [self._initial_dot_states()]
        seen = {self._dot_cycle_key(states[0]): 0}
        while True:
            nxt = self._step_dots(states[-1])
            key = self._dot_cycle_key(nxt)
            if key in seen:
                if seen[key] != 0:
                    raise RuntimeError("Dot cycle unexpectedly repeated before returning to start.")
                break
            states.append(nxt)
            seen[key] = len(states) - 1
        return states

    def _precompute_dot_bounds(self) -> List[Tuple[Tuple[float, float, float, float], ...]]:
        out: List[Tuple[Tuple[float, float, float, float], ...]] = []
        for phase_state in self.phase_dot_states:
            phase_bounds = tuple((dot.x + 10.0, dot.y + 10.0, 20.0, 20.0) for dot in phase_state)
            out.append(phase_bounds)
        return out

    @staticmethod
    def intersects(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw < bx or ax > bx + bw or ay + ah < by or ay > by + bh)

    @staticmethod
    def player_bounds(x: int, y: int) -> Tuple[float, float, float, float]:
        return (x - 15.0, y - 15.0, 31.0, 31.0)

    def get_state(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.phase)

    def _tile_at_snap(self, snap_x: int, snap_y: int) -> int | None:
        if 0 <= snap_x < self.tile_width and 0 <= snap_y < self.tile_height:
            return self.tile_types[snap_y][snap_x]
        return None

    def _get_relative_tile(self, x1: int, y1: int, x_off: int, y_off: int) -> int | None:
        snap_x = x1 // self.TILE_SIZE + x_off
        snap_y = y1 // self.TILE_SIZE + y_off
        return self._tile_at_snap(snap_x, snap_y)

    def _blocked_up(self, x: int, y: int) -> bool:
        a = self._get_relative_tile(x - 14, y + 24, 0, -1)
        b = self._get_relative_tile(x + 15, y + 24, 0, -1)
        return (a is not None and a == 0) or (b is not None and b == 0)

    def _blocked_down(self, x: int, y: int) -> bool:
        a = self._get_relative_tile(x - 14, y - 24, 0, 1)
        b = self._get_relative_tile(x + 15, y - 24, 0, 1)
        return (a is not None and a == 0) or (b is not None and b == 0)

    def _blocked_left(self, x: int, y: int) -> bool:
        a = self._get_relative_tile(x + 24, y - 15, -1, 0)
        b = self._get_relative_tile(x + 24, y + 14, -1, 0)
        return (a is not None and a == 0) or (b is not None and b == 0)

    def _blocked_right(self, x: int, y: int) -> bool:
        a = self._get_relative_tile(x - 24, y - 15, 1, 0)
        b = self._get_relative_tile(x - 24, y + 15, 1, 0)
        return (a is not None and a == 0) or (b is not None and b == 0)

    def is_goal_position(self, x: int, y: int) -> bool:
        pb = self.player_bounds(x, y)
        for goal_bound in self.goal_bounds:
            if self.intersects(pb, goal_bound):
                return True
        return False

    def collides_with_dot(self, x: int, y: int, phase: int) -> bool:
        pb = self.player_bounds(x, y)
        for dot_bound in self.dot_bounds_by_phase[phase]:
            if self.intersects(pb, dot_bound):
                return True
        return False

    def goal_distance(self, x: int, y: int) -> int:
        return min(abs(x - gx) + abs(y - gy) for gx, gy in self.goal_centers)

    def heuristic_to_goal(self, x: int, y: int) -> int:
        return self.goal_distance(x, y) // max(1, self.action_repeat)

    def _apply_movement(self, x: int, y: int, action: int) -> Tuple[int, int]:
        if action == self.ACTION_UP and not self._blocked_up(x, y):
            y -= 1
        elif action == self.ACTION_DOWN and not self._blocked_down(x, y):
            y += 1
        elif action == self.ACTION_LEFT and not self._blocked_left(x, y):
            x -= 1
        elif action == self.ACTION_RIGHT and not self._blocked_right(x, y):
            x += 1

        if x > self.BOARD_WIDTH:
            x = 0
        if x < 0:
            x = self.BOARD_WIDTH
        if y > self.BOARD_HEIGHT:
            y = 0
        if y < 0:
            y = self.BOARD_HEIGHT

        return x, y

    def micro_transition(self, state: Tuple[int, int, int], action: int) -> Tuple[Tuple[int, int, int], bool, bool]:
        x, y, phase = state
        next_phase = (phase + 1) % self.period
        x, y = self._apply_movement(x, y, action)
        dead = self.collides_with_dot(x, y, next_phase)
        win = self.is_goal_position(x, y)
        return (x, y, next_phase), dead, win

    def macro_transition(
        self, state: Tuple[int, int, int], action: int
    ) -> Tuple[Tuple[int, int, int], bool, bool, int]:
        cur = state
        dead = False
        win = False
        micro_steps = 0
        for _ in range(self.action_repeat):
            cur, dead, win = self.micro_transition(cur, action)
            micro_steps += 1
            if dead or win:
                break
        return cur, dead, win, micro_steps

    def reset(self) -> Tuple[int, int, int]:
        self.x, self.y = self.spawn
        self.phase = 0
        self.macro_steps = 0
        self.total_micro_steps = 0
        self.prev_goal_distance = self.goal_distance(self.x, self.y)
        self.done = False
        self.last_result = "running"
        return self.get_state()

    def snapshot(self) -> Dict[str, object]:
        dots = [(dot.x, dot.y) for dot in self.phase_dot_states[self.phase]]
        return {
            "player": (self.x, self.y),
            "phase": self.phase,
            "dots": dots,
            "macro_steps": self.macro_steps,
            "micro_steps": self.total_micro_steps,
            "result": self.last_result,
        }

    def step(self, action: int, capture_frames: bool = False):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        micro_frames: List[Dict[str, object]] = []
        dead = False
        win = False

        for _ in range(self.action_repeat):
            self.phase = (self.phase + 1) % self.period
            self.x, self.y = self._apply_movement(self.x, self.y, action)
            self.total_micro_steps += 1

            dead = self.collides_with_dot(self.x, self.y, self.phase)
            win = self.is_goal_position(self.x, self.y)

            if capture_frames:
                micro_frames.append(self.snapshot())

            if dead or win:
                break

        self.macro_steps += 1

        if dead:
            reward = self.death_penalty
            self.last_result = "death"
            self.done = self.terminate_on_death
            if not self.terminate_on_death:
                self.x, self.y = self.spawn
                self.phase = 0
                self.prev_goal_distance = self.goal_distance(self.x, self.y)
                self.last_result = "respawn"
        elif win:
            reward = self.goal_reward
            self.last_result = "win"
            self.done = True
        else:
            current_distance = self.goal_distance(self.x, self.y)
            progress = self.prev_goal_distance - current_distance
            reward = self.step_penalty + self.progress_reward_scale * progress
            self.prev_goal_distance = current_distance

            if self.macro_steps >= self.max_macro_steps:
                self.done = True
                self.last_result = "timeout"
            else:
                self.done = False
                self.last_result = "running"

        info = {
            "result": self.last_result,
            "macro_steps": self.macro_steps,
            "micro_steps": self.total_micro_steps,
            "frames": micro_frames,
        }
        return self.get_state(), reward, self.done, info
