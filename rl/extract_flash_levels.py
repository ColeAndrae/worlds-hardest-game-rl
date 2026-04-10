from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

NS = {"x": "http://ns.adobe.com/xfl/2008/"}

LEVEL_START_FRAME = 60
LEVEL_FRAME_STRIDE = 4
LEVEL_COUNT = 30
PLAYER_SYMBOL_ID = 154
COIN_SYMBOL_ID = 183


@dataclass
class Instance:
    symbol_id: int
    name: str
    tx: float
    ty: float
    script: str
    layer_name: str


@dataclass
class ConnectedComponent:
    pixels: np.ndarray
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract WHG Flash data into RL-friendly level files.")
    parser.add_argument(
        "--xfl",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "flash_xfl" / "the-worlds-hardest-g-1043817f" / "DOMDocument.xml",
        help="Path to DOMDocument.xml exported from the SWF.",
    )
    parser.add_argument(
        "--decompiled-root",
        type=Path,
        default=Path("/Users/cole/Desktop/worldshardestgame_decompiled"),
        help="Path containing 'frames/' and 'sprites/' exported by FFDec.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "flash_levels",
        help="Directory where extracted level data will be written.",
    )
    return parser.parse_args()


def to_bool_mask(image: np.ndarray, cond: np.ndarray) -> np.ndarray:
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[cond] = 1
    return mask


def connected_components(mask: np.ndarray, min_pixels: int) -> List[ConnectedComponent]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=np.uint8)
    out: List[ConnectedComponent] = []

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or seen[y, x]:
                continue

            stack = [(y, x)]
            seen[y, x] = 1
            pts: List[Tuple[int, int]] = []

            while stack:
                cy, cx = stack.pop()
                pts.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if mask[ny, nx] == 0 or seen[ny, nx]:
                        continue
                    seen[ny, nx] = 1
                    stack.append((ny, nx))

            if len(pts) < min_pixels:
                continue

            ys = np.array([p[0] for p in pts], dtype=np.int32)
            xs = np.array([p[1] for p in pts], dtype=np.int32)
            y0, y1 = int(ys.min()), int(ys.max())
            x0, x1 = int(xs.min()), int(xs.max())
            comp = ConnectedComponent(
                pixels=np.array([(x, y) for y, x in pts], dtype=np.int32),
                bbox=(x0, y0, x1, y1),
                center=(float(xs.mean()), float(ys.mean())),
            )
            out.append(comp)

    out.sort(key=lambda c: (c.center[1], c.center[0]))
    return out


def detect_blue_points(image: np.ndarray, min_pixels: int = 5) -> List[Tuple[float, float]]:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    blue = (b > 160) & (r < 90) & (g < 150)
    comps = connected_components(to_bool_mask(image, blue), min_pixels=min_pixels)
    return [comp.center for comp in comps]


def _flood_component(candidate: np.ndarray, sx: int, sy: int) -> np.ndarray:
    h, w = candidate.shape
    walk = np.zeros((h, w), dtype=bool)
    if not candidate[sy, sx]:
        return walk
    stack = [(sy, sx)]
    walk[sy, sx] = True
    while stack:
        cy, cx = stack.pop()
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            if walk[ny, nx] or not candidate[ny, nx]:
                continue
            walk[ny, nx] = True
            stack.append((ny, nx))
    return walk


def _nearest_candidate(candidate: np.ndarray, sx: int, sy: int, max_radius: int = 30) -> Tuple[int, int] | None:
    h, w = candidate.shape
    if candidate[sy, sx]:
        return (sx, sy)
    for radius in range(1, max_radius + 1):
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                nx = sx + dx
                ny = sy + dy
                if nx < 0 or ny < 0 or nx >= w or ny >= h:
                    continue
                if candidate[ny, nx]:
                    return (nx, ny)
    return None


def _best_component_from_candidate(
    candidate: np.ndarray,
    seed_xy: Tuple[float, float] | Sequence[Tuple[float, float]],
) -> np.ndarray:
    h, w = candidate.shape

    if isinstance(seed_xy, tuple) and len(seed_xy) == 2 and isinstance(seed_xy[0], (int, float)):
        seeds: List[Tuple[float, float]] = [(float(seed_xy[0]), float(seed_xy[1]))]
    else:
        seeds = [(float(x), float(y)) for x, y in seed_xy]

    if not seeds:
        return np.zeros((h, w), dtype=bool)

    best_walk = np.zeros((h, w), dtype=bool)
    best_count = -1

    for seed in seeds:
        sx = int(round(seed[0]))
        sy = int(round(seed[1]))
        sx = max(0, min(w - 1, sx))
        sy = max(0, min(h - 1, sy))

        nearest = _nearest_candidate(candidate, sx, sy, max_radius=35)
        if nearest is None:
            continue
        nx, ny = nearest
        walk = _flood_component(candidate, nx, ny)
        count = int(walk.sum())
        if count > best_count:
            best_count = count
            best_walk = walk

    return best_walk


def detect_walkable_mask(
    image: np.ndarray,
    seed_xy: Tuple[float, float] | Sequence[Tuple[float, float]],
) -> np.ndarray:
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)

    # Walls/solid borders are predominantly dark; walkable tiles are bright white/blue/green.
    dark = (r < 70) & (g < 70) & (b < 90)
    candidate = ~dark

    return _best_component_from_candidate(candidate, seed_xy)


def detect_walkable_mask_from_walls(
    walls: Sequence[Instance],
    sprites_dir: Path,
    stage_width: int,
    stage_height: int,
    seed_xy: Tuple[float, float] | Sequence[Tuple[float, float]],
) -> np.ndarray | None:
    blocked = np.zeros((stage_height, stage_width), dtype=bool)
    used = False

    for wall in walls:
        sprite_path = sprites_dir / f"DefineSprite_{wall.symbol_id}" / "1.png"
        if not sprite_path.exists():
            continue

        rgba = load_rgba(sprite_path)
        alpha = rgba[:, :, 3] > 0
        sh, sw = alpha.shape

        # Walls are center-registered in this SWF; place sprite alpha accordingly.
        ox = int(round(wall.tx - (sw / 2.0)))
        oy = int(round(wall.ty - (sh / 2.0)))

        x0 = max(0, ox)
        y0 = max(0, oy)
        x1 = min(stage_width, ox + sw)
        y1 = min(stage_height, oy + sh)
        if x1 <= x0 or y1 <= y0:
            continue

        sx0 = x0 - ox
        sy0 = y0 - oy
        sx1 = sx0 + (x1 - x0)
        sy1 = sy0 + (y1 - y0)
        blocked[y0:y1, x0:x1] |= alpha[sy0:sy1, sx0:sx1]
        used = True

    if not used:
        return None

    candidate = ~blocked
    return _best_component_from_candidate(candidate, seed_xy)


def carve_mask_around_points(mask: np.ndarray, points: Sequence[Tuple[float, float]], radius: int) -> np.ndarray:
    out = mask.copy()
    h, w = out.shape
    r = max(1, int(radius))
    for px, py in points:
        ix = int(round(px))
        iy = int(round(py))
        x0 = max(0, ix - r)
        x1 = min(w - 1, ix + r)
        y0 = max(0, iy - r)
        y1 = min(h - 1, iy + r)
        out[y0 : y1 + 1, x0 : x1 + 1] = True
    return out


def detect_green_components(image: np.ndarray) -> List[ConnectedComponent]:
    r = image[:, :, 0]
    g = image[:, :, 1]
    b = image[:, :, 2]
    green = (g > 140) & ((g - r) > 18) & ((g - b) > 15)
    return connected_components(to_bool_mask(image, green), min_pixels=80)


def load_rgb(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def load_rgba(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGBA"), dtype=np.uint8)


def list_sprite_frames(sprite_dir: Path) -> List[Path]:
    frames = []
    for p in sprite_dir.glob("*.png"):
        try:
            idx = int(p.stem)
        except ValueError:
            continue
        frames.append((idx, p))
    frames.sort(key=lambda t: t[0])
    return [p for _, p in frames]


def symbol_size(sprites_dir: Path, symbol_id: int) -> Tuple[int, int]:
    symbol_dir = sprites_dir / f"DefineSprite_{symbol_id}"
    first = symbol_dir / "1.png"
    if not first.exists():
        return (0, 0)
    with Image.open(first) as img:
        w, h = img.size
    return (int(w), int(h))


def parse_symbol_id(name: str) -> int:
    # "Symbol 154" -> 154
    return int(name.split()[1])


def parse_coins_required(script_text: str) -> List[int]:
    values = [int(m.group(1)) for m in re.finditer(r"coins\.push\((\d+)\)", script_text)]
    if len(values) != LEVEL_COUNT:
        raise RuntimeError(f"Expected {LEVEL_COUNT} coin entries, found {len(values)}.")
    return values


def parse_player_speed(script_text: str) -> float:
    m = re.search(r"var\s+speed\s*=\s*([0-9.]+)", script_text)
    if m:
        return float(m.group(1))
    return 3.0


def parse_win_check(script_text: str) -> str:
    for inc in re.finditer(r"_root\.currentLevel\+\+", script_text):
        prefix = script_text[: inc.start()]
        hits = list(re.finditer(r"hitTest\(_root\.(check\d)\)", prefix))
        if hits:
            return hits[-1].group(1)
    return "check2"


def parse_checkpoint_order(script_text: str) -> List[str]:
    checks = ["check1"]
    for m in re.finditer(r"_root\.currentCheck\s*=\s*_root\.(check\d)", script_text):
        c = m.group(1)
        if c not in checks:
            checks.append(c)
    return checks


def frame_instances_by_layer(timeline: ET.Element, frame_index: int) -> List[Instance]:
    out: List[Instance] = []
    layers = timeline.findall("x:layers/x:DOMLayer", NS)
    for layer in layers:
        layer_name = layer.attrib.get("name", "")
        frames_elem = layer.find("x:frames", NS)
        if frames_elem is None:
            continue

        active: ET.Element | None = None
        for frame in frames_elem.findall("x:DOMFrame", NS):
            start = int(frame.attrib.get("index", "0"))
            duration = int(frame.attrib.get("duration", "1"))
            if start <= frame_index < start + duration:
                active = frame

        if active is None:
            continue

        elems = active.find("x:elements", NS)
        if elems is None:
            continue

        for inst in elems.findall("x:DOMSymbolInstance", NS):
            lib_name = inst.attrib.get("libraryItemName", "")
            if not lib_name.startswith("Symbol "):
                continue
            sym_id = parse_symbol_id(lib_name)
            name = inst.attrib.get("name", "")

            matrix = inst.find("x:matrix/x:Matrix", NS)
            tx = float(matrix.attrib.get("tx", "0")) if matrix is not None else 0.0
            ty = float(matrix.attrib.get("ty", "0")) if matrix is not None else 0.0

            script_node = inst.find("x:Actionscript/x:script", NS)
            script = script_node.text or "" if script_node is not None else ""

            out.append(
                Instance(
                    symbol_id=sym_id,
                    name=name,
                    tx=tx,
                    ty=ty,
                    script=script,
                    layer_name=layer_name,
                )
            )
    return out


def script_layer_bootstrap(timeline: ET.Element) -> str:
    for layer in timeline.findall("x:layers/x:DOMLayer", NS):
        if layer.attrib.get("name") != "Script Layer":
            continue
        frames = layer.find("x:frames", NS)
        if frames is None:
            continue
        for frame in frames.findall("x:DOMFrame", NS):
            if int(frame.attrib.get("index", "0")) != 50:
                continue
            script_node = frame.find("x:Actionscript/x:script", NS)
            if script_node is not None:
                return script_node.text or ""
    raise RuntimeError("Failed to locate bootstrap script at Script Layer frame 50.")


def fit_translation(
    local_pts: Sequence[Tuple[float, float]],
    observed_pts: Sequence[Tuple[float, float]],
    tolerance: float = 4.0,
) -> Tuple[float, float, int, float]:
    if not local_pts or not observed_pts:
        return (0.0, 0.0, 0, 1e18)

    best_d = (0.0, 0.0)
    best_score = -1
    best_error = 1e18

    obs = np.array(observed_pts, dtype=np.float32)
    loc = np.array(local_pts, dtype=np.float32)

    for px, py in loc:
        for ox, oy in obs:
            dx = float(ox - px)
            dy = float(oy - py)
            shifted = loc + np.array([dx, dy], dtype=np.float32)
            # nearest-neighbor distance from shifted local to any observed point
            dist_sq = ((shifted[:, None, :] - obs[None, :, :]) ** 2).sum(axis=2)
            min_dist = np.sqrt(dist_sq.min(axis=1))
            matched = min_dist <= tolerance
            score = int(matched.sum())
            error = float(min_dist[matched].mean()) if score > 0 else 1e18

            if score > best_score or (score == best_score and error < best_error):
                best_score = score
                best_error = error
                best_d = (dx, dy)

    return (best_d[0], best_d[1], best_score, best_error)


def translate_points(points: Sequence[Tuple[float, float]], dx: float, dy: float) -> List[Tuple[float, float]]:
    return [(float(x + dx), float(y + dy)) for x, y in points]


def nearest_component(
    components: Sequence[ConnectedComponent],
    target_center: Tuple[float, float],
) -> ConnectedComponent | None:
    if not components:
        return None
    tx, ty = target_center
    best = min(components, key=lambda c: (c.center[0] - tx) ** 2 + (c.center[1] - ty) ** 2)
    return best


def component_bbox_to_dict(comp: ConnectedComponent) -> Dict[str, float]:
    x0, y0, x1, y1 = comp.bbox
    return {
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "cx": float(comp.center[0]),
        "cy": float(comp.center[1]),
    }


def point_distance(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def remove_matched(
    points: Sequence[Tuple[float, float]],
    matched: Sequence[Tuple[float, float]],
    tolerance: float = 4.0,
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for p in points:
        keep = True
        for q in matched:
            if point_distance(p, q) <= tolerance:
                keep = False
                break
        if keep:
            out.append(p)
    return out


def detect_colored_points_rgba(
    image: np.ndarray,
    min_pixels: int = 6,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    if image.shape[2] != 4:
        raise ValueError("detect_colored_points_rgba expects an RGBA image.")

    rgb = image[:, :, :3].astype(np.int16)
    alpha = image[:, :, 3]
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    intensity = rgb.sum(axis=2)

    # Keep colored, non-outline pixels from transparent sprites.
    mask = (alpha > 0) & (sat > 20) & (intensity > 60)
    comps = connected_components(to_bool_mask(image, mask), min_pixels=min_pixels)
    points = [comp.center for comp in comps]

    if np.any(mask):
        dominant = rgb[mask].mean(axis=0).astype(np.float32)
    else:
        dominant = np.array([0.0, 0.0, 255.0], dtype=np.float32)

    return points, dominant


def detect_points_near_color(
    image: np.ndarray,
    dominant_rgb: Sequence[float],
    min_pixels: int = 6,
) -> List[Tuple[float, float]]:
    r = image[:, :, 0].astype(np.int16)
    g = image[:, :, 1].astype(np.int16)
    b = image[:, :, 2].astype(np.int16)

    dr, dg, db = [float(v) for v in dominant_rgb]
    sat = np.maximum.reduce([r, g, b]) - np.minimum.reduce([r, g, b])

    # Specialized masks to avoid matching the board background for blue-heavy enemies.
    if db > dr + 40.0 and db > dg + 20.0:
        mask = (b > 140) & ((b - r) > 65) & ((b - g) > 40)
    elif dr > db + 40.0 and dr > dg + 20.0:
        mask = (r > 120) & ((r - g) > 35) & ((r - b) > 35)
    else:
        color = np.array([dr, dg, db], dtype=np.float32)
        diff = image.astype(np.float32) - color[None, None, :]
        dist = np.sqrt((diff**2).sum(axis=2))
        mask = (dist < 55.0) & (sat > 20)

    comps = connected_components(to_bool_mask(image, mask), min_pixels=min_pixels)
    return [comp.center for comp in comps]


def build_level_payload(
    level_num: int,
    level_frame_index: int,
    timeline: ET.Element,
    coins_required_all: Sequence[int],
    player_speed: float,
    frames_dir: Path,
    sprites_dir: Path,
    masks_dir: Path,
) -> Dict[str, object]:
    # Use the gameplay-render frame (label frame + 1), matching exported frame_<n+1> scripts.
    gameplay_frame = level_frame_index + 1
    frame_img_path = frames_dir / f"{gameplay_frame}.png"
    if not frame_img_path.exists():
        raise FileNotFoundError(f"Missing frame image: {frame_img_path}")
    frame_img = load_rgb(frame_img_path)
    h, w = frame_img.shape[:2]

    instances = frame_instances_by_layer(timeline, level_frame_index)
    player = next((i for i in instances if i.name == "player"), None)
    if player is None:
        raise RuntimeError(f"Level {level_num}: missing player instance.")

    walls = [i for i in instances if i.name == "walls"]
    enemies = [i for i in instances if i.name == "enemies"]
    checks = [i for i in instances if i.name.startswith("check")]
    coins = [i for i in instances if i.symbol_id == COIN_SYMBOL_ID and "currentCoins++" in i.script]

    player_size = symbol_size(sprites_dir, PLAYER_SYMBOL_ID)
    px = player.tx + player_size[0] / 2.0
    py = player.ty + player_size[1] / 2.0
    player_script = player.script
    win_check = parse_win_check(player_script)
    checkpoint_order = parse_checkpoint_order(player_script)

    blue_points = detect_blue_points(frame_img)

    # Enemy trajectories from sprite frame sequences, translated to match level-start points.
    enemy_tracks: List[List[List[float]]] = []
    enemy_start_points: List[Tuple[float, float]] = []
    for enemy in enemies:
        sprite_dir = sprites_dir / f"DefineSprite_{enemy.symbol_id}"
        sprite_frames = list_sprite_frames(sprite_dir)
        if not sprite_frames:
            continue

        local_tracks: List[List[Tuple[float, float]]] = []
        dominant_rgb = np.array([0.0, 0.0, 255.0], dtype=np.float32)
        for fp in sprite_frames:
            rgba = load_rgba(fp)
            pts, frame_dom = detect_colored_points_rgba(rgba, min_pixels=6)
            if pts:
                dominant_rgb = frame_dom
            # Fallback for sparse/edge-case sprites where the generic mask misses blue dots.
            if not pts:
                pts = detect_blue_points(rgba[:, :, :3], min_pixels=5)
            local_tracks.append(pts)

        if not local_tracks or not local_tracks[0]:
            continue

        observed_pts = detect_points_near_color(frame_img, dominant_rgb, min_pixels=6)
        if not observed_pts and dominant_rgb[2] > dominant_rgb[0] + 20.0:
            observed_pts = detect_blue_points(frame_img, min_pixels=5)

        dx, dy, score, _ = fit_translation(local_tracks[0], observed_pts, tolerance=6.0)
        min_required_match = max(1, int(0.5 * len(local_tracks[0])))
        if score < min_required_match:
            # Fallback to the instance transform from XFL if frame-based matching is unreliable.
            dx = enemy.tx
            dy = enemy.ty

        translated = [translate_points(frame_pts, dx, dy) for frame_pts in local_tracks]
        enemy_start_points.extend(translated[0])
        enemy_tracks.append(
            [[ [float(x), float(y)] for x, y in frame_pts ] for frame_pts in translated]
        )

    # If we detected no enemy track via sprites, fallback to static dots from the frame.
    if not enemy_tracks and blue_points:
        enemy_tracks = [[[ [float(x), float(y)] for x, y in blue_points ]]]
        enemy_start_points = list(blue_points)

    # Coin points:
    # Prefer scripted coin instances (exact, deterministic); fall back to frame-based subtraction.
    scripted_pts: List[Tuple[float, float]] = []
    if coins:
        for coin in coins:
            # Coin instances in this SWF use registration at the logical center.
            scripted_pts.append((coin.tx, coin.ty))

    required_coins = int(coins_required_all[level_num - 1])
    unmatched_blue = remove_matched(blue_points, enemy_start_points, tolerance=5.0)

    if scripted_pts:
        candidate_coin_points = list(scripted_pts)
        if len(candidate_coin_points) > required_coins and required_coins > 0:
            # Keep the scripted points that best align with non-enemy blue dots in the frame.
            scored = []
            for pt in candidate_coin_points:
                if unmatched_blue:
                    dist = min(point_distance(pt, ub) for ub in unmatched_blue)
                else:
                    dist = 1e9
                scored.append((dist, pt))
            scored.sort(key=lambda t: t[0])
            candidate_coin_points = [pt for _, pt in scored[:required_coins]]
    else:
        candidate_coin_points = unmatched_blue

    # Checkpoint regions are explicit symbol instances in the timeline; use those exact transforms.
    check_regions: Dict[str, Dict[str, float]] = {}
    for check in checks:
        sw, sh = symbol_size(sprites_dir, check.symbol_id)
        sw = max(1, int(sw))
        sh = max(1, int(sh))
        # Check symbols are center-registered (same coordinates used for respawn _x/_y).
        x0 = float(check.tx - (sw / 2.0))
        y0 = float(check.ty - (sh / 2.0))
        x1 = float(check.tx + (sw / 2.0) - 1.0)
        y1 = float(check.ty + (sh / 2.0) - 1.0)
        check_regions[check.name] = {
            "x0": float(max(0.0, min(float(w - 1), x0))),
            "y0": float(max(0.0, min(float(h - 1), y0))),
            "x1": float(max(0.0, min(float(w - 1), x1))),
            "y1": float(max(0.0, min(float(h - 1), y1))),
            "cx": float(check.tx),
            "cy": float(check.ty),
        }

    spawn = (px, py)
    if "check1" in check_regions:
        spawn = (check_regions["check1"]["cx"], check_regions["check1"]["cy"])

    if "check1" in check_regions:
        c1 = check_regions["check1"]
        walkable_seeds = [
            (float(c1["cx"]), float(c1["cy"])),
            (float(c1["x0"] + 2.0), float(c1["y0"] + 2.0)),
            (float(c1["x1"] - 2.0), float(c1["y0"] + 2.0)),
            (float(c1["x0"] + 2.0), float(c1["y1"] - 2.0)),
            (float(c1["x1"] - 2.0), float(c1["y1"] - 2.0)),
        ]
    else:
        walkable_seeds = [(px, py)]
    walkable = detect_walkable_mask_from_walls(
        walls=walls,
        sprites_dir=sprites_dir,
        stage_width=w,
        stage_height=h,
        seed_xy=walkable_seeds,
    )
    if walkable is None:
        walkable = detect_walkable_mask(frame_img, seed_xy=walkable_seeds)

    carve_points: List[Tuple[float, float]] = [(float(spawn[0]), float(spawn[1]))]
    carve_points.extend((float(v["cx"]), float(v["cy"])) for v in check_regions.values())
    carve_radius = max(10, int(round(player_size[0] / 2.0)) + 2)
    walkable = carve_mask_around_points(walkable, carve_points, radius=carve_radius)

    mask_path = masks_dir / f"level_{level_num:02d}_walkable.npz"
    np.savez_compressed(mask_path, walkable=walkable.astype(np.uint8))

    payload: Dict[str, object] = {
        "level_num": level_num,
        "frame_index": level_frame_index,
        "gameplay_frame_index": gameplay_frame,
        "resolution": {"width": int(w), "height": int(h)},
        "player_size": {"w": int(player_size[0]), "h": int(player_size[1])},
        "player_speed": float(player_speed),
        "spawn": {"x": float(spawn[0]), "y": float(spawn[1])},
        "checkpoint_order": checkpoint_order,
        "win_check": win_check,
        "check_regions": check_regions,
        "coins_required": required_coins,
        "coin_positions": [[float(x), float(y)] for x, y in candidate_coin_points],
        "enemy_tracks": enemy_tracks,
        "enemy_symbols": [enemy.symbol_id for enemy in enemies],
        "walls_symbols": [wall.symbol_id for wall in walls],
        "walkable_mask": str(mask_path.name),
    }
    return payload


def main() -> None:
    args = parse_args()

    dom_path = args.xfl
    decompiled_root = args.decompiled_root
    frames_dir = decompiled_root / "frames"
    sprites_dir = decompiled_root / "sprites"

    out_dir = args.out_dir
    levels_dir = out_dir / "levels"
    masks_dir = out_dir / "masks"
    out_dir.mkdir(parents=True, exist_ok=True)
    levels_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    if not dom_path.exists():
        raise FileNotFoundError(f"DOMDocument.xml not found: {dom_path}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames dir not found: {frames_dir}")
    if not sprites_dir.exists():
        raise FileNotFoundError(f"Sprites dir not found: {sprites_dir}")

    root = ET.parse(dom_path).getroot()
    timeline = root.find("x:timelines/x:DOMTimeline", NS)
    if timeline is None:
        raise RuntimeError("DOMTimeline not found in DOMDocument.xml")

    bootstrap_script = script_layer_bootstrap(timeline)
    coins_required_all = parse_coins_required(bootstrap_script)
    player_speed = parse_player_speed(bootstrap_script)

    index_payload: Dict[str, object] = {"levels": []}

    for level_num in range(1, LEVEL_COUNT + 1):
        frame_idx = LEVEL_START_FRAME + (level_num - 1) * LEVEL_FRAME_STRIDE
        payload = build_level_payload(
            level_num=level_num,
            level_frame_index=frame_idx,
            timeline=timeline,
            coins_required_all=coins_required_all,
            player_speed=player_speed,
            frames_dir=frames_dir,
            sprites_dir=sprites_dir,
            masks_dir=masks_dir,
        )

        level_json = levels_dir / f"level_{level_num:02d}.json"
        level_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        index_payload["levels"].append({"level_num": level_num, "file": str(level_json.name)})
        print(
            f"level {level_num:02d}: "
            f"coins={payload['coins_required']} "
            f"enemy_symbols={payload['enemy_symbols']} "
            f"checks={list(payload['check_regions'].keys())}"
        )

    (out_dir / "index.json").write_text(json.dumps(index_payload, indent=2), encoding="utf-8")
    print(f"wrote dataset: {out_dir}")


if __name__ == "__main__":
    main()
