import math

# ─── Constants ──────────────────────────────────────────────────────────────
SIM_W, SIM_H = 920, 620          # total window
PANEL_W       = 230               # right panel width
CANVAS_W      = SIM_W - PANEL_W  # simulation area
CANVAS_H      = SIM_H

# World dimensions in metres
WW, WH = 2.1, 1.4
LINE_W = 0.019                    # white line width (m)
GRID_RES = 0.006                  # spatial hash cell (m)

# Robot physical params
ROBOT_L = 0.20                    # robot length (m)
ROBOT_W = 0.14                    # robot width (m)
WHEEL_BASE = 0.13                 # distance between wheels (m)

# IR sensor array
N_IR      = 12
IR_SPACING = 0.009                 # center-to-center gap (m)
IR_SIZE    = 0.003                 # physical width of a sensor (m)
IR_FWDOFF  = 0.09                 # how far ahead of centre

DT = 1 / 200                     # simulation timestep (s)

# Colours
BG         = (30, 30, 30)
TRACK_COL  = (200, 200, 200)
LINE_COL   = (255, 255, 255)
ROBOT_IDLE = (70, 90, 110)
ROBOT_RUN  = (21, 101, 192)
ROBOT_OUT  = (186, 28, 28)
TRAIL_COL  = (66, 165, 245, 55)
IR_ON      = (239, 83, 80)
IR_OFF     = (102, 187, 106)
CAM_PT     = (255, 193, 7)
PANEL_BG   = (22, 22, 26)
PANEL_LINE = (50, 50, 55)
TEXT_PRI   = (230, 230, 230)
TEXT_SEC   = (140, 140, 150)
ACCENT     = (66, 165, 245)
GREEN_OK   = (46, 204, 113)
RED_ERR    = (231, 76, 60)
ORANGE     = (243, 156, 18)
PURPLE     = (171, 71, 188)
TEAL       = (38, 166, 154)

# ─── World → Canvas coordinate transform ────────────────────────────────────
MARGIN = 20
SCALE  = min((CANVAS_W - 2*MARGIN) / WW, (CANVAS_H - 2*MARGIN) / WH)

def w2c(x: float, y: float) -> tuple[int, int]:
    """World metres → canvas pixels."""
    cx = MARGIN + x * SCALE
    cy = MARGIN + y * SCALE
    return int(cx), int(cy)

def c2w(px: int, py: int) -> tuple[float, float]:
    return (px - MARGIN) / SCALE, (py - MARGIN) / SCALE
