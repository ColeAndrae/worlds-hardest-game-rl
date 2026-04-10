from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
try:
    from PIL import Image
except Exception:  # pragma: no cover - optional at runtime
    Image = None

State = Tuple[int, int, int, int, int]
PlannerState = Tuple[int, int, int]


@dataclass(frozen=True)
class CheckRegion:
    x0: float
    y0: float
    x1: float
    y1: float
    cx: float
    cy: float


class WorldsHardestGameFullEnv:
    """Headless environment driven by extracted Flash/XFL level data."""

    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4
    ACTION_UP_LEFT = 5
    ACTION_UP_RIGHT = 6
    ACTION_DOWN_LEFT = 7
    ACTION_DOWN_RIGHT = 8
    ACTIONS: Tuple[int, ...] = (
        ACTION_STAY,
        ACTION_UP,
        ACTION_DOWN,
        ACTION_LEFT,
        ACTION_RIGHT,
        ACTION_UP_LEFT,
        ACTION_UP_RIGHT,
        ACTION_DOWN_LEFT,
        ACTION_DOWN_RIGHT,
    )
    ACTION_NAMES: Dict[int, str] = {
        ACTION_STAY: "stay",
        ACTION_UP: "up",
        ACTION_DOWN: "down",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right",
        ACTION_UP_LEFT: "up_left",
        ACTION_UP_RIGHT: "up_right",
        ACTION_DOWN_LEFT: "down_left",
        ACTION_DOWN_RIGHT: "down_right",
    }
    _GLOBAL_REPAIRED_TRACK_CACHE: Dict[Tuple[int, int], List[np.ndarray]] = {}
    _GLOBAL_OBSERVED_ENEMY_CACHE: Dict[Tuple[int, int], np.ndarray] = {}
    MAX_LOCAL_POINTS: int = 640
    XFL_NS: Dict[str, str] = {"x": "http://ns.adobe.com/xfl/2008/"}

    def __init__(
        self,
        dataset_dir: str | Path | None = None,
        level_num: int = 1,
        action_repeat: int = 1,
        max_macro_steps: int = 1400,
        step_penalty: float = -0.002,
        progress_reward_scale: float = 0.004,
        coin_reward: float = 0.15,
        checkpoint_reward: float = 0.2,
        death_penalty: float = -1.0,
        goal_reward: float = 2.0,
        terminate_on_death: bool = True,
        enemy_hit_radius: float = 8.0,
        coin_collect_radius: float = 11.0,
        ignore_enemies: bool = False,
        transition_cache_limit: int = 1_000_000,
    ) -> None:
        self.dataset_dir = (
            Path(dataset_dir)
            if dataset_dir is not None
            else Path(__file__).resolve().parent / "data" / "flash_levels"
        )
        self.level_num = level_num
        self.action_repeat = int(action_repeat)
        self.max_macro_steps = int(max_macro_steps)

        self.step_penalty = float(step_penalty)
        self.progress_reward_scale = float(progress_reward_scale)
        self.coin_reward = float(coin_reward)
        self.checkpoint_reward = float(checkpoint_reward)
        self.death_penalty = float(death_penalty)
        self.goal_reward = float(goal_reward)
        self.terminate_on_death = bool(terminate_on_death)
        self.enemy_hit_radius = float(enemy_hit_radius)
        self.coin_collect_radius = float(coin_collect_radius)
        self.ignore_enemies = bool(ignore_enemies)
        self.transition_cache_limit = max(0, int(transition_cache_limit))

        self._micro_cache: Dict[Tuple[int, int, int, int], Tuple[PlannerState, bool]] = {}
        self._macro_cache: Dict[Tuple[int, int, int, int, int], Tuple[PlannerState, bool, int]] = {}
        self._xfl_symbol_cache: Dict[int, Dict[str, object]] = {}
        self._xfl_period_cache: Dict[int, int] = {}
        self._xfl_dom_timeline_cache: ET.Element | None = None
        self._xfl_level_enemy_cache: Dict[int, Dict[str, float] | None] = {}

        self.level_json_path = self.dataset_dir / "levels" / f"level_{self.level_num:02d}.json"
        self._load_level(self.level_num)
        self.reset()

    @staticmethod
    def _gcd(a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return abs(a)

    @classmethod
    def _lcm_cap(cls, values: Sequence[int], cap: int = 720) -> int:
        cur = 1
        for value in values:
            value = max(1, int(value))
            cur = (cur * value) // max(1, cls._gcd(cur, value))
            if cur > cap:
                return cap
        return max(1, cur)

    def _load_level(self, level_num: int) -> None:
        self.level_num = int(level_num)
        self.level_json_path = self.dataset_dir / "levels" / f"level_{self.level_num:02d}.json"
        if not self.level_json_path.exists():
            raise FileNotFoundError(f"Level JSON not found: {self.level_json_path}")

        payload = json.loads(self.level_json_path.read_text(encoding="utf-8"))
        self.level_frame_index = int(payload.get("frame_index", 0))

        res = payload["resolution"]
        self.width = int(res["width"])
        self.height = int(res["height"])

        ps = payload["player_size"]
        self.player_w = int(ps["w"])
        self.player_h = int(ps["h"])
        self.player_half_w = self.player_w / 2.0
        self.player_half_h = self.player_h / 2.0
        self._collision_offsets = np.array(
            [
                [0.0, 0.0],
                [self.player_half_w, 0.0],
                [-self.player_half_w, 0.0],
                [0.0, self.player_half_h],
                [0.0, -self.player_half_h],
                [self.player_half_w, self.player_half_h],
                [self.player_half_w, -self.player_half_h],
                [-self.player_half_w, self.player_half_h],
                [-self.player_half_w, -self.player_half_h],
            ],
            dtype=np.float32,
        )
        self._edge_x_offsets = np.linspace(
            -self.player_half_w,
            self.player_half_w,
            num=max(3, int(self.player_w) + 1),
            dtype=np.float32,
        )
        self._edge_y_offsets = np.linspace(
            -self.player_half_h,
            self.player_half_h,
            num=max(3, int(self.player_h) + 1),
            dtype=np.float32,
        )
        if self._edge_x_offsets.size == 0:
            self._edge_x_offsets = np.array([0.0], dtype=np.float32)
        if self._edge_y_offsets.size == 0:
            self._edge_y_offsets = np.array([0.0], dtype=np.float32)
        self.player_speed = float(payload.get("player_speed", 3.0))

        spawn = payload["spawn"]
        self.default_spawn = (float(spawn["x"]), float(spawn["y"]))

        self.checkpoint_order: List[str] = [str(v) for v in payload.get("checkpoint_order", ["check1"])]
        self.win_check = str(payload.get("win_check", "check2"))

        raw_regions = payload.get("check_regions", {})
        self.check_regions: Dict[str, CheckRegion] = {}
        for name, region in raw_regions.items():
            self.check_regions[str(name)] = CheckRegion(
                x0=float(region["x0"]),
                y0=float(region["y0"]),
                x1=float(region["x1"]),
                y1=float(region["y1"]),
                cx=float(region["cx"]),
                cy=float(region["cy"]),
            )

        if not self.checkpoint_order:
            self.checkpoint_order = ["check1"]
        if self.checkpoint_order[0] not in self.check_regions:
            self.check_regions[self.checkpoint_order[0]] = CheckRegion(
                x0=self.default_spawn[0] - 10.0,
                y0=self.default_spawn[1] - 10.0,
                x1=self.default_spawn[0] + 10.0,
                y1=self.default_spawn[1] + 10.0,
                cx=self.default_spawn[0],
                cy=self.default_spawn[1],
            )

        self.coins_required = int(payload.get("coins_required", 0))
        coin_positions = payload.get("coin_positions", [])
        if coin_positions:
            self.coin_positions = np.array(coin_positions, dtype=np.float32)
        else:
            self.coin_positions = np.zeros((0, 2), dtype=np.float32)

        enemy_tracks_payload = payload.get("enemy_tracks", [])
        self.enemy_symbols = [int(v) for v in payload.get("enemy_symbols", [])]
        self.enemy_tracks: List[List[np.ndarray]] = []
        periods: List[int] = []
        for track in enemy_tracks_payload:
            if not isinstance(track, list) or len(track) <= 0:
                continue

            frames: List[np.ndarray] = []
            for frame in track:
                frame_pts = self._normalize_enemy_frame(frame)
                frames.append(frame_pts)

            if not frames:
                continue
            self.enemy_tracks.append(frames)
            periods.append(len(frames))

        self.enemy_tracks = self._repair_enemy_tracks_from_xfl(self.enemy_tracks)
        periods = [len(track) for track in self.enemy_tracks if len(track) > 0]

        if not periods:
            periods = [1]
        self.phase_mod = self._lcm_cap(periods, cap=720)
        self._enemy_points_by_phase = self._precompute_enemy_points()

        mask_name = str(payload["walkable_mask"])
        mask_path = self.dataset_dir / "masks" / mask_name
        if not mask_path.exists():
            raise FileNotFoundError(f"Walkable mask missing: {mask_path}")
        self.walkable_mask = np.load(mask_path)["walkable"].astype(bool)
        if self.walkable_mask.shape[0] != self.height or self.walkable_mask.shape[1] != self.width:
            raise RuntimeError(
                f"Mask shape mismatch for level {self.level_num}: "
                f"{self.walkable_mask.shape} vs {(self.height, self.width)}"
            )

        self._reset_transition_caches()

    def _reset_transition_caches(self) -> None:
        self._micro_cache = {}
        self._macro_cache = {}

    def _xfl_library_dir(self) -> Path:
        return Path(__file__).resolve().parents[1] / "flash_xfl" / "the-worlds-hardest-g-1043817f" / "LIBRARY"

    def _xfl_dom_path(self) -> Path:
        return Path(__file__).resolve().parents[1] / "flash_xfl" / "the-worlds-hardest-g-1043817f" / "DOMDocument.xml"

    def _xfl_dom_timeline(self) -> ET.Element | None:
        cached = self._xfl_dom_timeline_cache
        if cached is not None:
            return cached
        dom_path = self._xfl_dom_path()
        if not dom_path.exists():
            return None
        root = ET.parse(dom_path).getroot()
        timeline = root.find("x:timelines/x:DOMTimeline", self.XFL_NS)
        if timeline is None:
            return None
        self._xfl_dom_timeline_cache = timeline
        return timeline

    def _candidate_frames_dirs(self) -> List[Path]:
        repo_root = Path(__file__).resolve().parents[1]
        home = Path.home()
        candidates = [
            repo_root / "flash_assets" / "worldshardestgame_decompiled" / "frames",
            repo_root / "worldshardestgame_decompiled" / "frames",
            home / "Desktop" / "projects" / "worldshardestgame_decompiled" / "frames",
            home / "Desktop" / "worldshardestgame_decompiled" / "frames",
            home / "Documents" / "Playground" / "worlds-hardest-game-rl-work" / "flash_assets" / "worldshardestgame_decompiled" / "frames",
        ]
        out: List[Path] = []
        seen: set[str] = set()
        for cand in candidates:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            if cand.exists() and cand.is_dir():
                out.append(cand)
        return out

    @staticmethod
    def _detect_blue_points_from_rgb(image: np.ndarray, min_pixels: int = 8) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] < 3:
            return np.zeros((0, 2), dtype=np.float32)
        r = image[:, :, 0]
        g = image[:, :, 1]
        b = image[:, :, 2]
        mask = (b > 155) & (r < 95) & (g < 145)
        h, w = mask.shape
        seen = np.zeros((h, w), dtype=np.uint8)
        points: List[Tuple[float, float]] = []
        for y in range(h):
            for x in range(w):
                if (not bool(mask[y, x])) or bool(seen[y, x]):
                    continue
                stack = [(y, x)]
                seen[y, x] = 1
                xs: List[int] = []
                ys: List[int] = []
                count = 0
                while stack:
                    cy, cx = stack.pop()
                    count += 1
                    xs.append(cx)
                    ys.append(cy)
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if (not bool(mask[ny, nx])) or bool(seen[ny, nx]):
                            continue
                        seen[ny, nx] = 1
                        stack.append((ny, nx))
                if count >= int(min_pixels):
                    points.append((float(np.mean(xs)), float(np.mean(ys))))
        if not points:
            return np.zeros((0, 2), dtype=np.float32)
        arr = np.asarray(points, dtype=np.float32).reshape((-1, 2))
        return arr

    def _observed_enemy_points_from_frame(self) -> np.ndarray:
        cache_key = (int(self.level_num), int(getattr(self, "level_frame_index", 0)))
        cached = self._GLOBAL_OBSERVED_ENEMY_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()
        if Image is None:
            return np.zeros((0, 2), dtype=np.float32)

        gameplay_frame = int(getattr(self, "level_frame_index", 0)) + 1
        points = np.zeros((0, 2), dtype=np.float32)
        for frames_dir in self._candidate_frames_dirs():
            frame_path = frames_dir / f"{gameplay_frame}.png"
            if not frame_path.exists():
                continue
            try:
                rgb = np.asarray(Image.open(frame_path).convert("RGB"), dtype=np.uint8)
            except Exception:
                continue
            points = self._detect_blue_points_from_rgb(rgb, min_pixels=8)
            if points.shape[0] <= 0:
                continue
            break

        if points.shape[0] > 0 and self.coin_positions.shape[0] > 0:
            keep: List[np.ndarray] = []
            for point in points:
                dist_sq = ((self.coin_positions - point[None, :]) ** 2).sum(axis=1)
                if float(np.min(dist_sq)) > 36.0:
                    keep.append(point)
            if keep:
                points = np.asarray(keep, dtype=np.float32).reshape((-1, 2))
            else:
                points = np.zeros((0, 2), dtype=np.float32)

        self._GLOBAL_OBSERVED_ENEMY_CACHE[cache_key] = points.astype(np.float32, copy=True)
        return points

    def _xfl_enemy_instance_meta(self, frame_index: int) -> Dict[str, float] | None:
        key = int(frame_index)
        if key in self._xfl_level_enemy_cache:
            return self._xfl_level_enemy_cache[key]

        timeline = self._xfl_dom_timeline()
        if timeline is None:
            self._xfl_level_enemy_cache[key] = None
            return None

        found: Dict[str, float] | None = None
        for layer in timeline.findall("x:layers/x:DOMLayer", self.XFL_NS):
            active: ET.Element | None = None
            for frame in layer.findall("x:frames/x:DOMFrame", self.XFL_NS):
                start = int(frame.attrib.get("index", "0"))
                duration = int(frame.attrib.get("duration", "1"))
                if start <= key < (start + duration):
                    active = frame
            if active is None:
                continue

            elems = active.find("x:elements", self.XFL_NS)
            if elems is None:
                continue
            for inst in elems.findall("x:DOMSymbolInstance", self.XFL_NS):
                if inst.attrib.get("name", "") != "enemies":
                    continue
                lib_name = inst.attrib.get("libraryItemName", "")
                if not lib_name.startswith("Symbol "):
                    continue
                symbol_id = int(lib_name.split()[1])
                matrix = inst.find("x:matrix/x:Matrix", self.XFL_NS)
                found = {
                    "symbol_id": float(symbol_id),
                    "first_frame": float(int(float(inst.attrib.get("firstFrame", "0")))),
                    "center_x": float(inst.attrib.get("centerPoint3DX", "0")),
                    "center_y": float(inst.attrib.get("centerPoint3DY", "0")),
                    "a": float(matrix.attrib.get("a", "1")) if matrix is not None else 1.0,
                    "b": float(matrix.attrib.get("b", "0")) if matrix is not None else 0.0,
                    "c": float(matrix.attrib.get("c", "0")) if matrix is not None else 0.0,
                    "d": float(matrix.attrib.get("d", "1")) if matrix is not None else 1.0,
                    "tx": float(matrix.attrib.get("tx", "0")) if matrix is not None else 0.0,
                    "ty": float(matrix.attrib.get("ty", "0")) if matrix is not None else 0.0,
                }
                break
            if found is not None:
                break

        self._xfl_level_enemy_cache[key] = found
        return found

    def _load_symbol_def(self, symbol_id: int) -> Dict[str, object]:
        symbol_id = int(symbol_id)
        cached = self._xfl_symbol_cache.get(symbol_id)
        if cached is not None:
            return cached

        symbol_path = self._xfl_library_dir() / f"Symbol {symbol_id}.xml"
        if not symbol_path.exists():
            data: Dict[str, object] = {"len": 1, "layers": [], "children": (), "has_instances": False}
            self._xfl_symbol_cache[symbol_id] = data
            return data

        root = ET.parse(symbol_path).getroot()
        layers_out: List[List[Tuple[int, int, List[Tuple[int, float, float, float, float, float, float, float | None, float | None]]]]] = []
        max_end = 1
        has_instances = False
        child_ids: set[int] = set()

        for layer in root.findall(".//x:DOMLayer", self.XFL_NS):
            frames_out: List[Tuple[int, int, List[Tuple[int, float, float, float, float, float, float, float | None, float | None]]]] = []
            for frame in layer.findall("./x:frames/x:DOMFrame", self.XFL_NS):
                start = int(frame.attrib.get("index", "0"))
                duration = int(frame.attrib.get("duration", "1"))
                max_end = max(max_end, start + duration)
                instances: List[Tuple[int, float, float, float, float, float, float, float | None, float | None]] = []

                elems = frame.find("x:elements", self.XFL_NS)
                if elems is not None:
                    for inst in elems.findall("x:DOMSymbolInstance", self.XFL_NS):
                        lib_name = inst.attrib.get("libraryItemName", "")
                        if not lib_name.startswith("Symbol "):
                            continue
                        child_id = int(lib_name.split()[1])
                        child_ids.add(child_id)

                        matrix = inst.find("x:matrix/x:Matrix", self.XFL_NS)
                        a = float(matrix.attrib.get("a", "1")) if matrix is not None else 1.0
                        b = float(matrix.attrib.get("b", "0")) if matrix is not None else 0.0
                        c = float(matrix.attrib.get("c", "0")) if matrix is not None else 0.0
                        d = float(matrix.attrib.get("d", "1")) if matrix is not None else 1.0
                        tx = float(matrix.attrib.get("tx", "0")) if matrix is not None else 0.0
                        ty = float(matrix.attrib.get("ty", "0")) if matrix is not None else 0.0

                        cx = inst.attrib.get("centerPoint3DX")
                        cy = inst.attrib.get("centerPoint3DY")
                        cx_f = float(cx) if cx is not None else None
                        cy_f = float(cy) if cy is not None else None
                        instances.append((child_id, a, b, c, d, tx, ty, cx_f, cy_f))

                if instances:
                    has_instances = True
                frames_out.append((start, duration, instances))
            if frames_out:
                layers_out.append(frames_out)

        data = {
            "len": int(max_end),
            "layers": layers_out,
            "children": tuple(sorted(child_ids)),
            "has_instances": bool(has_instances),
        }
        self._xfl_symbol_cache[symbol_id] = data
        return data

    @staticmethod
    def _active_frame_instances(
        frames: List[Tuple[int, int, List[Tuple[int, float, float, float, float, float, float, float | None, float | None]]]],
        t: int,
    ) -> List[Tuple[int, float, float, float, float, float, float, float | None, float | None]]:
        active: List[Tuple[int, float, float, float, float, float, float, float | None, float | None]] = []
        for start, duration, instances in frames:
            if start <= t < (start + duration):
                active = instances
        return active

    def _symbol_points_local(
        self,
        symbol_id: int,
        t: int,
        memo: Dict[Tuple[int, int, int], List[Tuple[float, float]]],
        use_center_delta: bool = False,
        depth: int = 0,
    ) -> List[Tuple[float, float]]:
        if depth > 12:
            return []

        symbol = self._load_symbol_def(symbol_id)
        symbol_len = max(1, int(symbol["len"]))
        symbol_period = max(symbol_len, self._symbol_period(symbol_id))
        t_mod = int(t) % symbol_len
        key = (int(symbol_id), int(t) % symbol_period, 1 if use_center_delta else 0)
        cached = memo.get(key)
        if cached is not None:
            return cached

        points: List[Tuple[float, float]] = []
        for layer_frames in symbol["layers"]:
            instances = self._active_frame_instances(layer_frames, t_mod)
            for child_id, a, b, c, d, tx, ty, cx, cy in instances:
                child = self._load_symbol_def(child_id)
                if bool(child["has_instances"]):
                    child_points = self._symbol_points_local(
                        child_id,
                        t,
                        memo,
                        use_center_delta=use_center_delta,
                        depth=depth + 1,
                    )
                    if use_center_delta and cx is not None and cy is not None and child_points:
                        child_base = self._symbol_points_local(
                            child_id,
                            0,
                            memo,
                            use_center_delta=use_center_delta,
                            depth=depth + 1,
                        )
                        if len(child_base) == len(child_points):
                            for (px, py), (bx, by) in zip(child_points, child_base):
                                qx = float(px) - float(bx)
                                qy = float(py) - float(by)
                                points.append((a * qx + c * qy + float(cx), b * qx + d * qy + float(cy)))
                            continue
                    for px, py in child_points:
                        points.append((a * px + c * py + tx, b * px + d * py + ty))
                else:
                    if cx is not None and cy is not None:
                        points.append((float(cx), float(cy)))
                    else:
                        points.append((float(tx), float(ty)))

        if len(points) > int(self.MAX_LOCAL_POINTS):
            idx = np.linspace(0, len(points) - 1, num=int(self.MAX_LOCAL_POINTS), dtype=np.int32)
            points = [points[int(i)] for i in idx.tolist()]

        memo[key] = points
        return points

    def _symbol_period(self, symbol_id: int, stack: set[int] | None = None) -> int:
        symbol_id = int(symbol_id)
        cached = self._xfl_period_cache.get(symbol_id)
        if cached is not None:
            return int(cached)

        if stack is None:
            stack = set()
        if symbol_id in stack:
            return max(1, int(self._load_symbol_def(symbol_id)["len"]))

        stack.add(symbol_id)
        symbol = self._load_symbol_def(symbol_id)
        values = [max(1, int(symbol["len"]))]
        for child_id in symbol["children"]:
            child = self._load_symbol_def(int(child_id))
            if bool(child["has_instances"]) or int(child["len"]) > 1:
                values.append(self._symbol_period(int(child_id), stack))
        stack.remove(symbol_id)
        period = self._lcm_cap(values, cap=720)
        self._xfl_period_cache[symbol_id] = int(period)
        return int(period)

    @staticmethod
    def _translation_score(
        local_pts: np.ndarray,
        observed_pts: np.ndarray,
        dx: float,
        dy: float,
        tolerance: float,
    ) -> Tuple[int, float]:
        if local_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return (0, 1e18)
        shifted = local_pts + np.array([dx, dy], dtype=np.float32)
        dist_sq = ((shifted[:, None, :] - observed_pts[None, :, :]) ** 2).sum(axis=2)
        dist = np.sqrt(dist_sq)

        # One-to-one greedy matching avoids over-counting many local points onto one observed point.
        pair_idx = np.argwhere(dist <= float(tolerance))
        if pair_idx.size == 0:
            return (0, 1e18)
        pair_d = dist[pair_idx[:, 0], pair_idx[:, 1]]
        order = np.argsort(pair_d, kind="mergesort")
        pair_idx = pair_idx[order]
        pair_d = pair_d[order]

        used_local = np.zeros((shifted.shape[0],), dtype=bool)
        used_observed = np.zeros((observed_pts.shape[0],), dtype=bool)
        matched_dist: List[float] = []
        for k in range(pair_idx.shape[0]):
            li = int(pair_idx[k, 0])
            oi = int(pair_idx[k, 1])
            if used_local[li] or used_observed[oi]:
                continue
            used_local[li] = True
            used_observed[oi] = True
            matched_dist.append(float(pair_d[k]))

        score = int(len(matched_dist))
        error = float(np.mean(np.asarray(matched_dist, dtype=np.float32))) if score > 0 else 1e18
        return (score, error)

    @staticmethod
    def _greedy_match_local_indices(
        local_pts: np.ndarray,
        observed_pts: np.ndarray,
        tolerance: float,
    ) -> List[int]:
        if local_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return []
        dist_sq = ((local_pts[:, None, :] - observed_pts[None, :, :]) ** 2).sum(axis=2)
        dist = np.sqrt(dist_sq)
        pair_idx = np.argwhere(dist <= float(tolerance))
        if pair_idx.size == 0:
            return []
        pair_d = dist[pair_idx[:, 0], pair_idx[:, 1]]
        order = np.argsort(pair_d, kind="mergesort")
        pair_idx = pair_idx[order]
        used_local = np.zeros((local_pts.shape[0],), dtype=bool)
        used_observed = np.zeros((observed_pts.shape[0],), dtype=bool)
        out: List[int] = []
        for k in range(pair_idx.shape[0]):
            li = int(pair_idx[k, 0])
            oi = int(pair_idx[k, 1])
            if used_local[li] or used_observed[oi]:
                continue
            used_local[li] = True
            used_observed[oi] = True
            out.append(li)
        return out

    @staticmethod
    def _subsample_points(points: np.ndarray, limit: int) -> np.ndarray:
        n = int(points.shape[0])
        if n <= limit:
            return points
        idx = np.linspace(0, n - 1, num=max(1, int(limit)), dtype=np.int32)
        return points[idx]

    @classmethod
    def _fit_translation_local_to_observed(
        cls,
        local_pts: np.ndarray,
        observed_pts: np.ndarray,
        tolerance: float = 6.0,
    ) -> Tuple[float, float, int, float]:
        if local_pts.shape[0] == 0 or observed_pts.shape[0] == 0:
            return (0.0, 0.0, 0, 1e18)

        seed_local = cls._subsample_points(local_pts, limit=32)
        seed_observed = cls._subsample_points(observed_pts, limit=32)

        candidates = (
            seed_observed[:, None, :] - seed_local[None, :, :]
        ).reshape((-1, 2))
        candidates = np.round(candidates * 2.0) / 2.0
        if candidates.shape[0] > 0:
            candidates = np.unique(candidates, axis=0)

        # Keep robust global guesses too, useful when candidate dedup is sparse.
        centroid = observed_pts.mean(axis=0) - local_pts.mean(axis=0)
        median = np.median(seed_observed, axis=0) - np.median(seed_local, axis=0)
        candidates = np.concatenate(
            [
                candidates,
                np.array([[centroid[0], centroid[1]], [median[0], median[1]]], dtype=np.float32),
            ],
            axis=0,
        )

        eval_local = cls._subsample_points(local_pts, limit=96)
        eval_observed = cls._subsample_points(observed_pts, limit=96)

        scored: List[Tuple[int, float, float, float]] = []
        for cand in candidates:
            dx = float(cand[0])
            dy = float(cand[1])
            score, error = cls._translation_score(eval_local, eval_observed, dx, dy, tolerance=tolerance)
            scored.append((score, error, dx, dy))

        scored.sort(key=lambda v: (-int(v[0]), float(v[1])))
        top = scored[: min(24, len(scored))]

        best_dx = 0.0
        best_dy = 0.0
        best_score = -1
        best_error = 1e18
        for _, _, dx, dy in top:
            score, error = cls._translation_score(local_pts, observed_pts, dx, dy, tolerance=tolerance)
            if score > best_score or (score == best_score and error < best_error):
                best_score = score
                best_error = error
                best_dx = dx
                best_dy = dy
        return (best_dx, best_dy, max(0, int(best_score)), float(best_error))

    def _repair_enemy_tracks_from_xfl(self, tracks: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
        if len(self.enemy_symbols) != 1:
            return tracks
        if not tracks or not tracks[0] or tracks[0][0].shape[0] == 0:
            return tracks

        observed = self._observed_enemy_points_from_frame()
        observed_from_payload_fallback = False
        if observed.shape[0] <= 0:
            if any(len(track) > 1 for track in tracks):
                return tracks
            observed = tracks[0][0].astype(np.float32, copy=False)
            observed_from_payload_fallback = True
        observed_n = int(observed.shape[0])
        symbol_id = int(self.enemy_symbols[0])
        cache_key = (int(getattr(self, "level_frame_index", 0)), int(symbol_id))
        cached_track = self._GLOBAL_REPAIRED_TRACK_CACHE.get(cache_key)
        if cached_track is not None:
            return [[frame.copy() for frame in cached_track]]
        enemy_meta = self._xfl_enemy_instance_meta(getattr(self, "level_frame_index", 0))
        if enemy_meta is not None:
            meta_symbol = int(enemy_meta.get("symbol_id", float(symbol_id)))
            if meta_symbol > 0:
                symbol_id = meta_symbol
                cache_key = (int(getattr(self, "level_frame_index", 0)), int(symbol_id))
                cached_track = self._GLOBAL_REPAIRED_TRACK_CACHE.get(cache_key)
                if cached_track is not None:
                    return [[frame.copy() for frame in cached_track]]

        period = self._symbol_period(symbol_id)
        if period <= 1:
            return tracks

        memo_raw: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}
        memo_center: Dict[Tuple[int, int, int], List[Tuple[float, float]]] = {}
        first_frame = int(enemy_meta.get("first_frame", 0.0)) if enemy_meta is not None else 0
        a = float(enemy_meta.get("a", 1.0)) if enemy_meta is not None else 1.0
        b = float(enemy_meta.get("b", 0.0)) if enemy_meta is not None else 0.0
        c = float(enemy_meta.get("c", 0.0)) if enemy_meta is not None else 0.0
        d = float(enemy_meta.get("d", 1.0)) if enemy_meta is not None else 1.0
        tx = float(enemy_meta.get("tx", 0.0)) if enemy_meta is not None else 0.0
        ty = float(enemy_meta.get("ty", 0.0)) if enemy_meta is not None else 0.0
        center_x = float(enemy_meta.get("center_x", tx)) if enemy_meta is not None else tx
        center_y = float(enemy_meta.get("center_y", ty)) if enemy_meta is not None else ty
        reg_x = center_x - tx
        reg_y = center_y - ty

        raw_local_frames: List[np.ndarray] = []
        center_local_frames: List[np.ndarray] = []
        for t in range(period):
            raw_pts = self._symbol_points_local(symbol_id, t, memo_raw, use_center_delta=False, depth=0)
            center_pts = self._symbol_points_local(symbol_id, t, memo_center, use_center_delta=True, depth=0)
            if raw_pts:
                raw_local_frames.append(np.array(raw_pts, dtype=np.float32).reshape((-1, 2)))
            else:
                raw_local_frames.append(np.zeros((0, 2), dtype=np.float32))
            if center_pts:
                center_local_frames.append(np.array(center_pts, dtype=np.float32).reshape((-1, 2)))
            else:
                center_local_frames.append(np.zeros((0, 2), dtype=np.float32))

        def transform_raw(frame: np.ndarray) -> np.ndarray:
            if frame.shape[0] == 0:
                return np.zeros((0, 2), dtype=np.float32)
            wx = a * frame[:, 0] + c * frame[:, 1] + tx
            wy = b * frame[:, 0] + d * frame[:, 1] + ty
            return np.stack([wx, wy], axis=1).astype(np.float32, copy=False)

        def transform_reg(frame: np.ndarray) -> np.ndarray:
            if frame.shape[0] == 0:
                return np.zeros((0, 2), dtype=np.float32)
            qx = frame[:, 0] - reg_x
            qy = frame[:, 1] - reg_y
            wx = a * qx + c * qy + tx
            wy = b * qx + d * qy + ty
            return np.stack([wx, wy], axis=1).astype(np.float32, copy=False)

        candidate_specs = [
            ("fit_raw", raw_local_frames, transform_raw, True),
            ("fit_raw_reg", raw_local_frames, transform_reg, True),
            ("fit_center_raw", center_local_frames, transform_raw, True),
            ("fit_center_reg", center_local_frames, transform_reg, True),
            ("exact_center_reg", center_local_frames, transform_reg, False),
        ]

        best_choice: Dict[str, object] | None = None
        offset_bias = int(first_frame) % max(1, period)
        for name, local_frames, transform_fn, allow_shift in candidate_specs:
            active_offsets = [i for i, fr in enumerate(local_frames) if fr.shape[0] > 0]
            if not active_offsets:
                continue
            if len(active_offsets) > 480:
                step = max(1, len(active_offsets) // 480)
                active_offsets = active_offsets[::step]
            if offset_bias not in active_offsets and 0 <= offset_bias < len(local_frames):
                active_offsets = [offset_bias] + active_offsets

            candidate_best: Dict[str, object] | None = None
            for offset in active_offsets:
                transformed = transform_fn(local_frames[offset])
                if transformed.shape[0] == 0:
                    continue
                if allow_shift:
                    dx, dy, score, error = self._fit_translation_local_to_observed(transformed, observed, tolerance=6.0)
                else:
                    score, error = self._translation_score(transformed, observed, 0.0, 0.0, tolerance=6.0)
                    dx, dy = (0.0, 0.0)

                pick = {
                    "name": name,
                    "offset": int(offset),
                    "dx": float(dx),
                    "dy": float(dy),
                    "score": int(score),
                    "error": float(error),
                    "count_diff": int(abs(int(transformed.shape[0]) - observed_n)),
                    "shift_allowed": bool(allow_shift),
                    "local_frames": local_frames,
                    "transform_fn": transform_fn,
                }
                if candidate_best is None:
                    candidate_best = pick
                    continue
                if int(pick["score"]) > int(candidate_best["score"]):
                    candidate_best = pick
                    continue
                if int(pick["score"]) == int(candidate_best["score"]):
                    if int(pick["count_diff"]) < int(candidate_best["count_diff"]):
                        candidate_best = pick
                        continue
                    if int(pick["count_diff"]) == int(candidate_best["count_diff"]) and float(pick["error"]) < float(
                        candidate_best["error"]
                    ):
                        candidate_best = pick
                        continue
                    if float(pick["error"]) < float(candidate_best["error"]):
                        candidate_best = pick
                        continue
                    if (
                        float(pick["error"]) == float(candidate_best["error"])
                        and abs(int(pick["offset"]) - offset_bias) < abs(int(candidate_best["offset"]) - offset_bias)
                    ):
                        candidate_best = pick
            if candidate_best is None:
                continue

            if best_choice is None:
                best_choice = candidate_best
                continue
            if int(candidate_best["score"]) > int(best_choice["score"]):
                best_choice = candidate_best
                continue
            if int(candidate_best["score"]) == int(best_choice["score"]):
                # Prefer exact/no-shift when equally good; otherwise lower fit error.
                if bool(best_choice["shift_allowed"]) and (not bool(candidate_best["shift_allowed"])):
                    best_choice = candidate_best
                    continue
                if int(candidate_best["count_diff"]) < int(best_choice["count_diff"]):
                    best_choice = candidate_best
                    continue
                if float(candidate_best["error"]) < float(best_choice["error"]):
                    best_choice = candidate_best
                    continue

        if best_choice is None:
            return tracks
        min_score = max(1, int(0.18 * max(1, observed_n)))
        if int(best_choice["score"]) < min_score:
            return tracks

        chosen_frames: List[np.ndarray] = best_choice["local_frames"]  # type: ignore[assignment]
        chosen_transform = best_choice["transform_fn"]  # type: ignore[assignment]
        chosen_offset = int(best_choice["offset"])
        chosen_dx = float(best_choice["dx"])
        chosen_dy = float(best_choice["dy"])

        keep_local_idx: np.ndarray | None = None
        start_local = chosen_frames[chosen_offset]
        if start_local.shape[0] > 0 and observed_n > 0:
            start_world = chosen_transform(start_local)
            if start_world.shape[0] > 0:
                start_world = (start_world + np.array([chosen_dx, chosen_dy], dtype=np.float32)).astype(
                    np.float32, copy=False
                )
                matched_idx = self._greedy_match_local_indices(start_world, observed, tolerance=6.0)
                local_n = int(start_world.shape[0])
                matched_n = int(len(matched_idx))

                # Only prune when local symbolic points clearly explode far beyond observed enemies.
                # This avoids dropping legitimate enemy dots on near-matching levels.
                should_prune = (
                    local_n >= max(24, int(math.ceil(2.0 * float(observed_n))))
                    and matched_n >= max(6, int(math.ceil(0.45 * float(observed_n))))
                )
                payload_mismatch_prune = (
                    observed_from_payload_fallback
                    and local_n > observed_n
                    and matched_n >= max(1, int(math.ceil(0.65 * float(observed_n))))
                )
                if payload_mismatch_prune or should_prune:
                    keep_local_idx = np.asarray(sorted(set(int(v) for v in matched_idx)), dtype=np.int32)

        repaired_frames: List[np.ndarray] = []
        non_empty = 0
        for out_t in range(period):
            local_t = (chosen_offset + out_t) % period
            local_frame = chosen_frames[local_t]
            if local_frame.shape[0] == 0:
                repaired_frames.append(np.zeros((0, 2), dtype=np.float32))
                continue
            transformed = chosen_transform(local_frame)
            if transformed.shape[0] == 0:
                repaired_frames.append(np.zeros((0, 2), dtype=np.float32))
                continue
            shifted = (transformed + np.array([chosen_dx, chosen_dy], dtype=np.float32)).astype(np.float32, copy=False)
            if keep_local_idx is not None and shifted.shape[0] > 0:
                valid_idx = keep_local_idx[keep_local_idx < shifted.shape[0]]
                if valid_idx.size > 0:
                    shifted = shifted[valid_idx]
            repaired_frames.append(shifted)
            non_empty += 1

        if non_empty <= 0:
            return tracks
        self._GLOBAL_REPAIRED_TRACK_CACHE[cache_key] = [frame.copy() for frame in repaired_frames]
        return [repaired_frames]

    def _precompute_enemy_points(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for phase in range(self.phase_mod):
            pts: List[np.ndarray] = []
            for track in self.enemy_tracks:
                idx = phase % len(track)
                frame_pts = track[idx]
                if frame_pts.shape[0] > 0:
                    pts.append(frame_pts)
            if pts:
                out.append(np.concatenate(pts, axis=0).astype(np.float32))
            else:
                out.append(np.zeros((0, 2), dtype=np.float32))
        return out

    @staticmethod
    def _normalize_enemy_frame(frame: object) -> np.ndarray:
        arr = np.asarray(frame, dtype=np.float32)
        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if arr.ndim == 1:
            if arr.size % 2 != 0:
                return np.zeros((0, 2), dtype=np.float32)
            arr = arr.reshape((-1, 2))
            return arr.astype(np.float32, copy=False)

        if arr.ndim >= 2 and arr.shape[-1] == 2:
            arr = arr.reshape((-1, 2))
            return arr.astype(np.float32, copy=False)

        flat = arr.reshape((-1,))
        if flat.size % 2 != 0:
            return np.zeros((0, 2), dtype=np.float32)
        arr = flat.reshape((-1, 2))
        return arr.astype(np.float32, copy=False)

    @staticmethod
    def _snap_point(point: Tuple[float, float]) -> Tuple[float, float]:
        return (float(int(round(point[0]))), float(int(round(point[1]))))

    def set_level(self, level_num: int) -> State:
        self._load_level(level_num)
        return self.reset()

    def _sample_player_points(self, x: float, y: float) -> Tuple[Tuple[float, float], ...]:
        samples = self._collision_offsets + np.array([x, y], dtype=np.float32)
        return tuple((float(px), float(py)) for px, py in samples.tolist())

    def _is_walkable_xy(self, x: float, y: float) -> bool:
        ix = int(math.floor(x + 0.5))
        iy = int(math.floor(y + 0.5))
        if ix < 0 or iy < 0 or ix >= self.width or iy >= self.height:
            return False
        return bool(self.walkable_mask[iy, ix])

    def _blocked_up(self, x: float, y: float) -> bool:
        probe_y = y - self.player_half_h
        for ox in self._edge_x_offsets:
            if not self._is_walkable_xy(x + float(ox), probe_y):
                return True
        return False

    def _blocked_down(self, x: float, y: float) -> bool:
        probe_y = y + self.player_half_h
        for ox in self._edge_x_offsets:
            if not self._is_walkable_xy(x + float(ox), probe_y):
                return True
        return False

    def _blocked_left(self, x: float, y: float) -> bool:
        probe_x = x - self.player_half_w
        for oy in self._edge_y_offsets:
            if not self._is_walkable_xy(probe_x, y + float(oy)):
                return True
        return False

    def _blocked_right(self, x: float, y: float) -> bool:
        probe_x = x + self.player_half_w
        for oy in self._edge_y_offsets:
            if not self._is_walkable_xy(probe_x, y + float(oy)):
                return True
        return False

    def _script_move_delta(self, action: int) -> Tuple[float, float]:
        speed = float(self.player_speed)
        dx = 0.0
        dy = 0.0
        if action in (self.ACTION_LEFT, self.ACTION_UP_LEFT, self.ACTION_DOWN_LEFT):
            dx -= speed
        elif action in (self.ACTION_RIGHT, self.ACTION_UP_RIGHT, self.ACTION_DOWN_RIGHT):
            dx += speed
        if action in (self.ACTION_UP, self.ACTION_UP_LEFT, self.ACTION_UP_RIGHT):
            dy -= speed
        elif action in (self.ACTION_DOWN, self.ACTION_DOWN_LEFT, self.ACTION_DOWN_RIGHT):
            dy += speed
        return dx, dy

    def _apply_movement_flash_script(self, x: float, y: float, action: int) -> Tuple[float, float]:
        # Match original Flash script semantics:
        # 1) apply directional speed once per frame
        # 2) correct against walls using four point hit-tests.
        dx, dy = self._script_move_delta(action)
        nx = x + dx
        ny = y + dy
        speed = float(self.player_speed)

        if not self._is_walkable_xy(nx + self.player_half_w - speed, ny):
            nx -= speed
        if not self._is_walkable_xy(nx - self.player_half_w + speed, ny):
            nx += speed
        if not self._is_walkable_xy(nx, ny - self.player_half_h + speed):
            ny += speed
        if not self._is_walkable_xy(nx, ny + self.player_half_h - speed):
            ny -= speed
        return nx, ny

    def _apply_movement(self, x: float, y: float, action: int) -> Tuple[float, float]:
        return self._apply_movement_flash_script(x, y, action)

    def _apply_movement_with_collision(self, x: float, y: float, action: int, phase: int) -> Tuple[float, float, bool]:
        nx, ny = self._apply_movement_flash_script(x, y, action)
        dead = self.collides_with_enemy(nx, ny, phase)
        return nx, ny, dead

    def _enemy_points(self, phase: int) -> np.ndarray:
        return self._enemy_points_by_phase[int(phase) % self.phase_mod]

    def _collides_with_enemy_points(self, x: float, y: float, enemies: np.ndarray) -> bool:
        if self.ignore_enemies or enemies.shape[0] == 0:
            return False
        # Fatal on any overlap between player AABB and enemy circle.
        rx0 = x - self.player_half_w
        rx1 = x + self.player_half_w
        ry0 = y - self.player_half_h
        ry1 = y + self.player_half_h
        rr_sq = float(self.enemy_hit_radius * self.enemy_hit_radius)
        ex = enemies[:, 0]
        ey = enemies[:, 1]
        cx = np.clip(ex, rx0, rx1)
        cy = np.clip(ey, ry0, ry1)
        dx = ex - cx
        dy = ey - cy
        dist_sq = dx * dx + dy * dy
        return bool(np.any(dist_sq <= rr_sq))

    def collides_with_enemy(self, x: float, y: float, phase: int) -> bool:
        enemies = self._enemy_points(phase)
        return self._collides_with_enemy_points(x, y, enemies)

    @staticmethod
    def _inside(region: CheckRegion, x: float, y: float) -> bool:
        return (region.x0 <= x <= region.x1) and (region.y0 <= y <= region.y1)

    def _update_checkpoint(self, x: float, y: float) -> bool:
        advanced = False
        for idx, name in enumerate(self.checkpoint_order):
            if idx <= self.current_checkpoint_idx:
                continue
            region = self.check_regions.get(name)
            if region is None:
                continue
            if self._inside(region, x, y):
                self.current_checkpoint_idx = idx
                self.current_checkpoint_name = name
                self.current_spawn = self._snap_point((region.cx, region.cy))
                advanced = True
        return advanced

    def _try_collect_coins(self, x: float, y: float) -> int:
        if self.coin_positions.shape[0] == 0:
            return 0
        remaining = np.where(~self.coin_collected)[0]
        if remaining.size == 0:
            return 0

        points = self.coin_positions[remaining]
        diff = points - np.array([x, y], dtype=np.float32)[None, :]
        dist_sq = (diff**2).sum(axis=1)
        hit_local = np.where(dist_sq <= (self.coin_collect_radius**2))[0]
        if hit_local.size == 0:
            return 0

        hit_global = remaining[hit_local]
        self.coin_collected[hit_global] = True
        return int(hit_global.size)

    @property
    def coins_collected_count(self) -> int:
        return int(self.coin_collected.sum())

    @property
    def coins_remaining_count(self) -> int:
        return int(self.coin_collected.shape[0] - self.coin_collected.sum())

    def remaining_coin_positions(self) -> np.ndarray:
        if self.coin_positions.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.float32)
        return self.coin_positions[~self.coin_collected]

    def win_center(self) -> Tuple[float, float]:
        region = self.check_regions.get(self.win_check)
        if region is None:
            return self.default_spawn
        return (region.cx, region.cy)

    def _distance_to_objective(self, x: float, y: float) -> float:
        remaining = self.remaining_coin_positions()
        if remaining.shape[0] > 0:
            diff = remaining - np.array([x, y], dtype=np.float32)[None, :]
            dist = np.sqrt((diff**2).sum(axis=1))
            return float(dist.min())
        wx, wy = self.win_center()
        return float(math.hypot(x - wx, y - wy))

    def _is_win(self, x: float, y: float) -> bool:
        if self.coins_collected_count < self.coins_required:
            return False
        region = self.check_regions.get(self.win_check)
        if region is None:
            return False
        return self._inside(region, x, y)

    def _respawn(self) -> None:
        self.x, self.y = self.current_spawn
        self.coin_collected[:] = False
        self.prev_objective_distance = self._distance_to_objective(self.x, self.y)

    def get_state(self) -> State:
        phase_state = 0 if self.ignore_enemies else int(self.phase)
        return (
            int(round(self.x)),
            int(round(self.y)),
            phase_state,
            int(self.current_checkpoint_idx),
            int(self.coins_collected_count),
        )

    def get_planner_state(self) -> PlannerState:
        phase_state = 0 if self.ignore_enemies else int(self.phase)
        return (int(round(self.x)), int(round(self.y)), phase_state)

    def micro_transition_from(self, state: PlannerState, action: int) -> Tuple[PlannerState, bool]:
        x, y, phase = state
        cache_key = (int(x), int(y), int(phase), int(action))
        cached = self._micro_cache.get(cache_key)
        if cached is not None:
            return cached

        next_phase = 0 if self.ignore_enemies else (phase + 1) % self.phase_mod
        nx, ny, dead = self._apply_movement_with_collision(float(x), float(y), action, next_phase)
        out = ((int(round(nx)), int(round(ny)), int(next_phase)), dead)
        if self.transition_cache_limit <= 0 or len(self._micro_cache) < self.transition_cache_limit:
            self._micro_cache[cache_key] = out
        return out

    def macro_transition_from(
        self,
        state: PlannerState,
        action: int,
        repeat: int | None = None,
    ) -> Tuple[PlannerState, bool, int]:
        x, y, phase = state
        steps = int(self.action_repeat if repeat is None else repeat)
        cache_key = (int(x), int(y), int(phase), int(action), steps)
        cached = self._macro_cache.get(cache_key)
        if cached is not None:
            return cached

        cur = state
        micro_steps = 0
        dead = False
        for _ in range(steps):
            cur, dead = self.micro_transition_from(cur, action)
            micro_steps += 1
            if dead:
                break
        out = (cur, dead, micro_steps)
        if self.transition_cache_limit <= 0 or len(self._macro_cache) < self.transition_cache_limit:
            self._macro_cache[cache_key] = out
        return out

    def reset(self, level_num: int | None = None) -> State:
        if level_num is not None and int(level_num) != self.level_num:
            self._load_level(int(level_num))

        self.current_checkpoint_idx = 0
        self.current_checkpoint_name = self.checkpoint_order[0]
        region = self.check_regions.get(self.current_checkpoint_name)
        if region is not None:
            self.current_spawn = self._snap_point((region.cx, region.cy))
        else:
            self.current_spawn = self._snap_point(self.default_spawn)

        self.x, self.y = self.current_spawn
        self.phase = 0
        self.macro_steps = 0
        self.total_micro_steps = 0
        self.deaths = 0
        self.coin_collected = np.zeros((self.coin_positions.shape[0],), dtype=bool)

        self.prev_objective_distance = self._distance_to_objective(self.x, self.y)
        self.done = False
        self.last_result = "running"
        return self.get_state()

    def snapshot(self) -> Dict[str, object]:
        enemies = self._enemy_points(self.phase)
        return {
            "level_num": self.level_num,
            "ignore_enemies": bool(self.ignore_enemies),
            "enemy_hit_radius": float(self.enemy_hit_radius),
            "player": (float(self.x), float(self.y)),
            "phase": int(self.phase),
            "enemies": enemies.astype(np.float32).tolist(),
            "coins": [
                (float(p[0]), float(p[1]), bool(collected))
                for p, collected in zip(self.coin_positions.tolist(), self.coin_collected.tolist())
            ],
            "macro_steps": int(self.macro_steps),
            "micro_steps": int(self.total_micro_steps),
            "deaths": int(self.deaths),
            "coins_collected": int(self.coins_collected_count),
            "coins_required": int(self.coins_required),
            "result": self.last_result,
        }

    def step(self, action: int, capture_frames: bool = False):
        if self.done:
            raise RuntimeError("Episode is done. Call reset() before step().")

        total_reward = 0.0
        micro_frames: List[Dict[str, object]] = []
        done_this_step = False

        for _ in range(self.action_repeat):
            self.phase = (self.phase + 1) % self.phase_mod
            self.x, self.y, dead = self._apply_movement_with_collision(self.x, self.y, action, self.phase)
            self.total_micro_steps += 1

            reward = self.step_penalty
            gained = self._try_collect_coins(self.x, self.y)
            if gained > 0:
                reward += self.coin_reward * float(gained)

            checkpoint_advanced = self._update_checkpoint(self.x, self.y)
            if checkpoint_advanced:
                reward += self.checkpoint_reward

            win = self._is_win(self.x, self.y)

            if dead:
                self.deaths += 1
                reward += self.death_penalty
                if self.terminate_on_death:
                    self.done = True
                    self.last_result = "death"
                else:
                    self._respawn()
                    self.last_result = "respawn"
                done_this_step = True
            elif win:
                reward += self.goal_reward
                self.done = True
                self.last_result = "win"
                done_this_step = True
            else:
                current_dist = self._distance_to_objective(self.x, self.y)
                progress = self.prev_objective_distance - current_dist
                reward += self.progress_reward_scale * progress
                self.prev_objective_distance = current_dist
                self.last_result = "running"

            total_reward += reward

            if capture_frames:
                micro_frames.append(self.snapshot())

            if done_this_step:
                break

        self.macro_steps += 1
        if not self.done and self.macro_steps >= self.max_macro_steps:
            self.done = True
            self.last_result = "timeout"

        info = {
            "result": self.last_result,
            "macro_steps": int(self.macro_steps),
            "micro_steps": int(self.total_micro_steps),
            "deaths": int(self.deaths),
            "coins_collected": int(self.coins_collected_count),
            "coins_required": int(self.coins_required),
            "frames": micro_frames,
        }
        return self.get_state(), total_reward, self.done, info
