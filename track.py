import random
import math
import numpy as np
from typing import Optional
from dataclasses import dataclass
from config import *


# ─── Track generation ────────────────────────────────────────────────────────

def chaikin(pts: np.ndarray, iters: int = 1) -> np.ndarray:
    for _ in range(iters):
        new = []
        n = len(pts)
        for i in range(n):
            a, b = pts[i], pts[(i + 1) % n]
            new.append(0.75*a + 0.25*b)
            new.append(0.25*a + 0.75*b)
        pts = np.array(new)
    return pts

def densify(pts: np.ndarray, step: float = 0.008) -> np.ndarray:
    result = []
    n = len(pts)
    for i in range(n):
        a, b = pts[i], pts[(i + 1) % n]
        dist  = np.linalg.norm(b - a)
        steps = max(1, int(dist / step))
        for j in range(steps):
            t = j / steps
            result.append(a + t*(b - a))
    return np.array(result)

def gen_track(use_sharp: bool = True) -> np.ndarray:
    """Generate a random closed track in world coordinates."""
    rng   = random.Random()
    N     = rng.randint(7, 12)
    angles = sorted([2*math.pi*i/N + rng.uniform(-0.5, 0.5)*(2*math.pi/N*0.7)
                     for i in range(N)])
    noise = 0.06 if use_sharp else 0.03
    pts   = np.array([[0.5 + (0.30+rng.random()*0.16)*math.cos(a) + rng.uniform(-noise, noise),
                       0.5 + (0.19+rng.random()*0.13)*math.sin(a) + rng.uniform(-noise, noise)]
                      for a in angles])
    iters = 4 if use_sharp else 7
    pts   = chaikin(pts, iters)
    # Normalise to world box with margins
    mg    = 0.32
    mn, mx = pts.min(0), pts.max(0)
    sw, sh = WW - 2*mg, WH - 2*mg
    pts   = mg + (pts - mn) / (mx - mn) * np.array([sw, sh])
    return densify(pts)

def gen_hard_track_1() -> np.ndarray:
    pts = []
    sc = 0.6
    def add_arc(cx, cy, r, a_start, a_end):
        dist = abs(a_end - a_start) * r
        steps = max(2, int(dist / (0.05*sc)))
        for i in range(steps + 1):
            a = a_start + (a_end - a_start) * i / steps
            pts.append([cx + r*math.cos(a), cy + r*math.sin(a)])
            
    # Start: line heading right
    pts.append([0.5*sc, 0.5*sc])
    pts.append([2.5*sc, 0.5*sc])
    # 15cm radius curve U-turn (0.15 * 0.6 = 0.09)
    add_arc(2.5*sc, 0.65*sc, 0.15*sc, -math.pi/2, math.pi/2)
    # Heading left
    pts.append([1.8*sc, 0.8*sc])
    # Very sharp angle (< 90 deg) -> heading up/right
    pts.append([1.2*sc, 0.8*sc])
    pts.append([1.7*sc, 1.2*sc]) 
    # Another sharp angle -> heading left/down
    pts.append([0.8*sc, 1.1*sc])
    # Curve to point back down
    add_arc(0.8*sc, 1.4*sc, 0.3*sc, -math.pi/2, -math.pi)
    # Heading down
    pts.append([0.5*sc, 1.4*sc])
    pts.append([0.5*sc, 0.5*sc]) # back to start
    
    return densify(np.array(pts))

def gen_hard_track_2() -> np.ndarray:
    pts = []
    sc = 0.6
    def add_arc(cx, cy, r, a_start, a_end):
        dist = abs(a_end - a_start) * r
        steps = max(2, int(dist / (0.05*sc)))
        for i in range(steps + 1):
            a = a_start + (a_end - a_start) * i / steps
            pts.append([cx + r*math.cos(a), cy + r*math.sin(a)])
            
    # Zig-zags on the left, then straight to the right, then tight curves
    pts.append([0.6*sc, 0.4*sc])
    pts.append([2.6*sc, 0.4*sc])
    add_arc(2.6*sc, 0.55*sc, 0.15*sc, -math.pi/2, math.pi/2)
    pts.append([2.0*sc, 0.7*sc])
    add_arc(2.0*sc, 0.85*sc, 0.15*sc, -math.pi/2, -3*math.pi/2)
    pts.append([2.6*sc, 1.0*sc])
    add_arc(2.6*sc, 1.15*sc, 0.15*sc, -math.pi/2, math.pi/2)
    
    # move left to zig-zags
    pts.append([1.0*sc, 1.3*sc])
    # <90 deg zigzag
    pts.append([0.6*sc, 1.3*sc])
    pts.append([1.2*sc, 1.6*sc])
    pts.append([0.4*sc, 1.6*sc])
    # curve back
    add_arc(0.4*sc, 1.0*sc, 0.6*sc, math.pi/2, 3*math.pi/2)
    pts.append([0.4*sc, 0.4*sc])
    
    return densify(np.array(pts))

def gen_hard_track() -> np.ndarray:
    return gen_hard_track_1() if random.random() < 0.5 else gen_hard_track_2()

@dataclass
class TrackFeatures:
    gap_start: Optional[int]  = None   # index where gap begins
    gap_end:   Optional[int]  = None
    fp_pts:    Optional[np.ndarray] = None   # false path segment
    finish_idx: int = 0                # finish line index

def add_features(track: np.ndarray,
                 use_gap: bool, use_fp: bool) -> TrackFeatures:
    n   = len(track)
    f   = TrackFeatures()
    # Find straight sections by measuring heading change
    straight = []
    for i in range(5, n - 5):
        p0, p1, p2 = track[(i-4) % n], track[i], track[(i+4) % n]
        a1  = math.atan2(*(p1 - p0)[::-1])
        a2  = math.atan2(*(p2 - p1)[::-1])
        da  = abs(a2 - a1)
        if da > math.pi:
            da = 2*math.pi - da
        if da < 0.05:
            straight.append(i)

    if use_gap and len(straight) > 10:
        idx       = straight[len(straight)//4]
        gap_len   = int(0.10 / 0.008)
        f.gap_start = idx
        f.gap_end   = min(idx + gap_len, n - 1)

    if use_fp and len(straight) > 10:
        idx  = straight[int(len(straight)*0.70)]
        pt   = track[idx]
        nxt  = track[(idx + 5) % n]
        d    = nxt - pt
        d   /= np.linalg.norm(d)
        perp = np.array([-d[1], d[0]])
        flen = 0.095  # slightly under 10cm fake branch limit
        end  = pt + perp * flen
        # densify false path
        s    = max(1, int(flen / 0.006))
        fp   = np.array([pt + perp * flen * j / s for j in range(s+1)])
        f.fp_pts = fp

    f.finish_idx = int(n * 0.88)
    return f

# ─── Spatial grid for fast line detection ───────────────────────────────────

class LineGrid:
    def __init__(self, track: np.ndarray, features: TrackFeatures):
        cols = int(WW / GRID_RES) + 2
        rows = int(WH / GRID_RES) + 2
        self.cols = cols
        self.rows = rows
        self.grid = np.zeros((rows, cols), dtype=np.uint8)
        hw = max(1, int((LINE_W/2) / GRID_RES))

        def stamp(x: float, y: float):
            gc = int(x / GRID_RES)
            gr = int(y / GRID_RES)
            for dr in range(-hw, hw+1):
                for dc in range(-hw, hw+1):
                    if dr*dr + dc*dc <= hw*hw + hw:
                        r, c = gr+dr, gc+dc
                        if 0 <= r < rows and 0 <= c < cols:
                            self.grid[r, c] = 1

        n = len(track)
        gs, ge = features.gap_start, features.gap_end
        for i, pt in enumerate(track):
            if gs is not None and gs <= i <= ge:
                continue
            stamp(pt[0], pt[1])

        if features.fp_pts is not None:
            for pt in features.fp_pts:
                stamp(pt[0], pt[1])

        # Double finish line (two thin stripes perpendicular to track)
        fi = features.finish_idx
        pt  = track[fi]
        nxt = track[(fi + 1) % n]
        d   = nxt - pt
        d  /= np.linalg.norm(d)
        perp = np.array([-d[1], d[0]])
        mw  = 0.09
        for off in [-0.020, 0.020]:
            base = pt + d * off
            for k in range(50):
                t   = (k/49 - 0.5) * 2 * mw
                stamp(base[0] + perp[0]*t, base[1] + perp[1]*t)

    def on_line(self, x: float, y: float) -> bool:
        c = int(x / GRID_RES)
        r = int(y / GRID_RES)
        if 0 <= r < self.rows and 0 <= c < self.cols:
            return bool(self.grid[r, c])
        return False
