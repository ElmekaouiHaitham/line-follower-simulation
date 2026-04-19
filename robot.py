import math
from dataclasses import dataclass
import numpy as np

from config import *

# ─── Robot model ─────────────────────────────────────────────────────────────

class Robot:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.vl = 0.0   # actual left wheel speed
        self.vr = 0.0   # actual right wheel speed

        self.tidx = 0
        self.base_speed = 0.5
        self.state = "FOLLOWING"   # kept for display in main.py
        self.pid = PID()

        # Gap logic
        self.t = 0.0
        self.gap_start_t = 0.0
        self.GAP_TIMEOUT = 0.5    # seconds to coast straight over a gap

    def update(self, ir_data, cam_err, dt, mass, target_baseline):
        """
        Main update loop called every simulation step.
        ir_data: (err, lost, readings)
        """
        self.t += dt
        ir_err, ir_lost, ir_readings = ir_data if ir_data else (0.0, True, [])

        self.base_speed = target_baseline

        # ─── Gap / lost-line logic ────────────────────────────────────────────
        if not ir_lost:
            # Line visible — normal PID
            self.state = "FOLLOWING"
            self.gap_start_t = self.t          # reset gap timer
            corr = self.pid.compute(ir_err, dt)
            vl_cmd = self.base_speed - corr
            vr_cmd = self.base_speed + corr
        elif self.t - self.gap_start_t < self.GAP_TIMEOUT:
            # Line lost but within timeout — coast straight
            self.state = "GAP_NAV"
            vl_cmd, vr_cmd = self.base_speed, self.base_speed
        else:
            # Timeout exceeded — stop
            self.state = "LOST"
            vl_cmd, vr_cmd = 0.0, 0.0

        self.step_physics(vl_cmd, vr_cmd, dt, mass)

    def step_physics(self, vl_cmd, vr_cmd, dt, mass):
        # First-order lag on motor response (GA12-N20 simulation)
        tau = 0.04 + mass * 0.10
        a   = dt / (tau + dt)
        self.vl += a * (vl_cmd - self.vl)
        self.vr += a * (vr_cmd - self.vr)

        v     = (self.vl + self.vr) / 2
        omega = (self.vr - self.vl) / WHEEL_BASE
        self.theta += omega * dt
        self.x     += v * math.cos(self.theta) * dt
        self.y     += v * math.sin(self.theta) * dt


def make_robot(track: np.ndarray) -> Robot:
    p   = track[0]
    nxt = track[min(5, len(track) - 1)]
    ang = math.atan2(nxt[1] - p[1], nxt[0] - p[0])
    return Robot(x=p[0], y=p[1], theta=ang)


def update_tidx(r: Robot, track: np.ndarray):
    n    = len(track)
    best = r.tidx
    bd   = float('inf')
    for di in range(-5, 25):
        idx = (r.tidx + di) % n
        dx  = track[idx][0] - r.x
        dy  = track[idx][1] - r.y
        d   = dx * dx + dy * dy
        if d < bd:
            bd   = d
            best = idx
    r.tidx = best


# ─── PID controller ──────────────────────────────────────────────────────────

@dataclass
class PID:
    kp: float = 4.5
    ki: float = 0.0
    kd: float = 0.60
    _I: float = 0.0
    _pe: float = 0.0

    def compute(self, err: float, dt: float) -> float:
        self._I  = max(-0.5, min(0.5, self._I + err * dt))
        D        = (err - self._pe) / dt
        self._pe = err
        return self.kp * err + self.ki * self._I + self.kd * D

    def reset(self):
        self._I  = 0.0
        self._pe = 0.0
