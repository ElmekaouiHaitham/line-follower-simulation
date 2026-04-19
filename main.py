"""
Le Fil d'Ariane — Line Follower Robot Simulator
================================================
Controls:
  SPACE        Play / Pause
  R            Reset robot (same track)
  N / H        New track / Hard track
  1 / 2 / 3    Switch sensor mode  (1=IR, 2=Camera, 3=Hybrid)
  +/-          Simulation speed  (1x / 3x / 8x)
  Mouse        Drag sliders in the panel
  ESC / Q      Quit
"""

import os
import pygame
import sys
import math
import random
import numpy as np
from typing import Optional, List, Tuple

# Import modularized components
from config import *
from track import gen_track, gen_hard_track, add_features, LineGrid, TrackFeatures
from robot import Robot, make_robot, update_tidx, PID
from ui import Slider, Button, ErrGraph, draw_stat
from algorithms.ir import read_ir
from algorithms.camera import read_camera
from algorithms.hybrid import compute_hybrid_error


# ─── Main simulation class ────────────────────────────────────────────────────

class Sim:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("Le Fil d'Ariane — Robot Simulator")
        self.screen = pygame.display.set_mode((SIM_W, SIM_H))
        self.clock  = pygame.time.Clock()

        # Surfaces
        self.canvas_surf = pygame.Surface((CANVAS_W, CANVAS_H))
        self.panel_surf  = pygame.Surface((PANEL_W,  SIM_H))
        self.trail_surf  = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
        self.graph_surf  = pygame.Surface((PANEL_W - 20, 70))

        # Fonts
        self.fn_lg  = pygame.font.SysFont("monospace", 16, bold=True)
        self.fn_md  = pygame.font.SysFont("monospace", 13, bold=True)
        self.fn_sm  = pygame.font.SysFont("monospace", 11)
        self.fn_xs  = pygame.font.SysFont("monospace",  9)
        self.fn_ttl = pygame.font.SysFont("sans-serif", 14, bold=True)

        # Sim state
        self.running   = False
        self.sim_mult  = 1          # 1, 3, or 8
        self.mode      = 'IR'       # IR / CAM / HYB
        self.noise     = False
        self.use_gap   = True
        self.use_fp    = True
        self.use_sharp = True

        self.track    : Optional[np.ndarray]   = None
        self.features : Optional[TrackFeatures] = None
        self.grid     : Optional[LineGrid]      = None
        self.robot    : Optional[Robot]         = None
        self.err_graph = ErrGraph()
        self.trail_pts : List[Tuple[int,int]] = []

        self.stats = dict(lap_t=0.0, best=None, laps=0, err=0.0,
                          spd=0.0, pen=0)
        self.accum  = 0.0
        self.last_ir : tuple = None
        self.last_cam: tuple = None

        # Sliders
        self._build_sliders()

        # Buttons (drawn in panel space — offset applied at draw time)
        self._build_buttons()

        # Generate first track
        os.makedirs("laps", exist_ok=True)
        self.new_track()

    # ── UI building ───────────────────────────────────────────────────────────

    def _build_sliders(self):
        sx = 12
        self.sl_spd = Slider("Speed (m/s)",  0.10, 1.50, 0.50, ".2f", sx, 100)
        self.sl_mass= Slider("Mass (kg)",    0.10, 1.00, 0.30, ".2f", sx, 130)
        self.sl_look= Slider("Cam look (m)", 0.10, 0.60, 0.30, ".2f", sx, 160)
        self.sl_kp  = Slider("Kp",           0.0,  18.0, 4.50, ".1f", sx, 200)
        self.sl_ki  = Slider("Ki",           0.0,   3.0, 0.00, ".2f", sx, 228)
        self.sl_kd  = Slider("Kd",           0.0,   6.0, 0.60, ".2f", sx, 256)
        self.sliders = [self.sl_spd, self.sl_mass, self.sl_look,
                        self.sl_kp,  self.sl_ki,   self.sl_kd]

    def _build_buttons(self):
        bw, bh, bsx = 62, 24, 4
        y_mode = 40
        self.btn_ir  = Button("IR",     12,           y_mode, bw, bh, ACCENT)
        self.btn_cam = Button("Camera", 12+bw+bsx,    y_mode, bw, bh, PURPLE)
        self.btn_hyb = Button("Hybrid", 12+2*(bw+bsx),y_mode, bw, bh, TEAL)
        self.btn_ir.active = True

        y2 = 280
        bw2 = (PANEL_W - 28) // 2
        self.btn_play  = Button("Play",       12,       y2,    bw2, 28, GREEN_OK)
        self.btn_reset = Button("Reset",      12+bw2+4, y2,    bw2, 28, ORANGE)
        self.btn_new   = Button("Random Track",12,      y2+36, bw2, 24, (80,80,90))
        self.btn_hard  = Button("Hard Track", 12+bw2+4, y2+36, bw2, 24, (200,80,80))
        self.btn_autop = Button("Auto PID",   12,       y2+68, PANEL_W-24, 24, (80,80,90))
        self.btn_noise = Button("Noise: OFF", 12,       y2+100,PANEL_W-24, 24, (80,80,90))

        self.mode_buttons = [self.btn_ir, self.btn_cam, self.btn_hyb]
        self.all_buttons  = (self.mode_buttons +
                             [self.btn_play, self.btn_reset,
                              self.btn_new, self.btn_hard, self.btn_autop, self.btn_noise])

    # ── Track lifecycle ───────────────────────────────────────────────────────

    def new_track(self, hard_mode=False):
        self.running  = False
        self.btn_play.label  = "Play"
        self.btn_play.active = False
        if hard_mode:
            self.track = gen_hard_track()
        else:
            self.track = gen_track(self.use_sharp)
        self.features = add_features(self.track, self.use_gap, self.use_fp)
        self.grid     = LineGrid(self.track, self.features)
        self.reset_robot()
        self.err_graph= ErrGraph()
        # Pre-render track into a surface
        self._render_track_bg()

    def reset_robot(self):
        self.robot     = make_robot(self.track)
        self.robot.pid.reset()
        self.trail_pts = []
        self.trail_surf.fill((0, 0, 0, 0))
        self.stats     = dict(lap_t=0.0, best=None, laps=0, err=0.0,
                              spd=0.0, pen=0)
        self.accum     = 0.0
        self.err_graph.clear()
        self.last_ir   = None
        self.last_cam  = None

    # ── Track background surface ──────────────────────────────────────────────

    def _render_track_bg(self):
        self.track_bg = pygame.Surface((CANVAS_W, CANVAS_H))
        self.track_bg.fill(BG)
        # Concrete texture (subtle dots)
        for _ in range(600):
            px = random.randint(0, CANVAS_W-1)
            py = random.randint(0, CANVAS_H-1)
            v  = random.randint(34, 40)
            self.track_bg.set_at((px, py), (v, v, v))

        t   = self.track
        f   = self.features
        n   = len(t)
        lw  = max(2, int(LINE_W * SCALE))
        gs, ge = f.gap_start, f.gap_end

        # Draw track line segment by segment
        for i in range(n):
            if gs is not None and gs <= i < ge:
                continue
            a  = w2c(*t[i])
            b  = w2c(*t[(i+1) % n])
            pygame.draw.line(self.track_bg, LINE_COL, a, b, lw)

        # Gap indicator
        if gs is not None:
            mid = (gs + ge) // 2
            px, py = w2c(*t[mid])
            pygame.draw.line(self.track_bg, ORANGE,
                             w2c(*t[gs]), w2c(*t[ge]), 1)
            lbl = self.fn_xs.render("GAP 10cm", True, ORANGE)
            self.track_bg.blit(lbl, (px+6, py-10))

        # False path
        if f.fp_pts is not None:
            pts = [w2c(*p) for p in f.fp_pts]
            if len(pts) > 1:
                pygame.draw.lines(self.track_bg, LINE_COL, False, pts, lw)
            mid_p = pts[len(pts)//2]
            lbl = self.fn_xs.render("FALSE PATH", True, RED_ERR)
            self.track_bg.blit(lbl, (mid_p[0]+4, mid_p[1]-12))

        # Finish line markers (double white line)
        fi  = f.finish_idx
        pt  = t[fi]
        nxt = t[(fi+1) % n]
        d   = nxt - pt
        d  /= np.linalg.norm(d)
        perp = np.array([-d[1], d[0]])
        mw   = 0.09
        for off in [-0.020, 0.020]:
            base = pt + d*off
            a = w2c(*(base - perp*mw))
            b = w2c(*(base + perp*mw))
            pygame.draw.line(self.track_bg, LINE_COL, a, b, lw)
        fl = self.fn_xs.render("FINISH", True, (150, 150, 150))
        fp = w2c(*(pt + perp*0.12))
        self.track_bg.blit(fl, fp)

        # Start marker
        sp  = w2c(*t[0])
        sl  = self.fn_xs.render("START", True, (100, 200, 100))
        self.track_bg.blit(sl, (sp[0]+4, sp[1]-12))

    # ── Sim step ──────────────────────────────────────────────────────────────

    def sim_step(self):
        r, g = self.robot, self.grid
        ir_data = cam_data = None
        err = 0.0

        if self.mode in ('IR', 'HYB'):
            ir_err, ir_lost, ir_rd = read_ir(r, g, self.noise)
            ir_data = (ir_err, ir_lost, ir_rd)
            err = ir_err

        if self.mode in ('CAM', 'HYB'):
            cam_err, cam_lost, cam_pts = read_camera(r, g, self.sl_look.value)
            cam_data = (cam_err, cam_lost, cam_pts)
            if self.mode == 'CAM':
                err = cam_err
            else:
                err = compute_hybrid_error(ir_data, cam_err)

        # Sync PID params to robot's internal controller
        r.pid.kp = self.sl_kp.value
        r.pid.ki = self.sl_ki.value
        r.pid.kd = self.sl_kd.value

        # Robot internal update handles state transitions and movement
        r.update(ir_data, err, DT, self.sl_mass.value, self.sl_spd.value)
        update_tidx(r, self.track)

        self.stats['lap_t'] += DT
        self.stats['err']    = err
        self.stats['spd']    = (r.vl + r.vr) / 2
        self.stats['state']  = r.state

        # Penalty: IR fully lost while in IR-only mode
        if ir_data and ir_data[1] and self.mode == 'IR':
            if random.random() < 0.0008:
                self.stats['pen'] += 5

        # Finish detection
        fi  = self.features.finish_idx
        dp  = math.hypot(r.x - self.track[fi][0], r.y - self.track[fi][1])
        if dp < 0.10 and self.stats['lap_t'] > 2.0:
            lt = self.stats['lap_t']
            if self.stats['best'] is None or lt < self.stats['best']:
                self.stats['best'] = lt
            
            # Save lap graph
            self.stats['laps'] += 1
            fname = f"laps/lap_{self.stats['laps']}.png"
            self.err_graph.save_to_file(fname, self.fn_xs)
            self.err_graph.clear()
            
            self.stats['lap_t']  = 0.0
            self.robot.pid.reset()

        # Trail
        cp = w2c(r.x, r.y)
        self.trail_pts.append(cp)
        if len(self.trail_pts) > 900:
            self.trail_pts.pop(0)
            self.trail_surf.fill((0, 0, 0, 0))
            if len(self.trail_pts) > 1:
                pygame.draw.lines(self.trail_surf, (66, 165, 245, 45),
                                  False, self.trail_pts, 2)

        # Error graph
        self.err_graph.push(err)
        self.last_ir  = ir_data
        self.last_cam = cam_data

    # ── Drawing ───────────────────────────────────────────────────────────────

    def draw_canvas(self):
        self.canvas_surf.blit(self.track_bg, (0, 0))

        # Trail
        if len(self.trail_pts) > 1:
            pygame.draw.lines(self.canvas_surf, (66, 165, 245, 80),
                              False, self.trail_pts, 2)

        r = self.robot
        if r is None:
            return

        # Camera detected points
        if self.last_cam and len(self.last_cam[2]) > 0:
            for pt in self.last_cam[2]:
                cp = w2c(pt[0], pt[1])
                pygame.draw.circle(self.canvas_surf, CAM_PT, cp, 2)

        # Draw robot
        cx, cy = w2c(r.x, r.y)
        rl = int(ROBOT_L * SCALE)
        rw = int(ROBOT_W * SCALE)

        # Camera lookahead cone
        if self.mode in ('CAM', 'HYB'):
            look_px = int(self.sl_look.value * SCALE)
            cone_surf = pygame.Surface((CANVAS_W, CANVAS_H), pygame.SRCALPHA)
            ang_half  = math.pi / 3.5
            pts_cone  = [(cx, cy)]
            steps = 20
            for i in range(steps + 1):
                a = r.theta - ang_half + 2*ang_half*i/steps
                pts_cone.append((int(cx + math.cos(a)*look_px),
                                 int(cy + math.sin(a)*look_px)))
            pygame.draw.polygon(cone_surf, (*CAM_PT, 22), pts_cone)
            self.canvas_surf.blit(cone_surf, (0, 0))
            # Cone edges
            for sign in [-1, 1]:
                ea = r.theta + sign*ang_half
                ex = int(cx + math.cos(ea)*look_px)
                ey = int(cy + math.sin(ea)*look_px)
                pygame.draw.line(self.canvas_surf, (*CAM_PT, 80), (cx, cy), (ex, ey), 1)

        # Robot body (rotated rectangle via polygon)
        cos_t, sin_t = math.cos(r.theta), math.sin(r.theta)
        corners = [
            (-rl//2, -rw//2), ( rl//2, -rw//2),
            ( rl//2,  rw//2), (-rl//2,  rw//2),
        ]
        color = ROBOT_RUN if self.running else ROBOT_IDLE
        poly  = [(int(cx + c[0]*cos_t - c[1]*sin_t),
                  int(cy + c[0]*sin_t + c[1]*cos_t)) for c in corners]
        pygame.draw.polygon(self.canvas_surf, color, poly)
        pygame.draw.polygon(self.canvas_surf, ACCENT, poly, 1)

        # Direction arrow
        ax = int(cx + math.cos(r.theta)*(rl//2 + 8))
        ay = int(cy + math.sin(r.theta)*(rl//2 + 8))
        pygame.draw.circle(self.canvas_surf, ACCENT, (ax, ay), 4)

        # IR sensors
        if self.last_ir:
            ir_rad = max(2, int(IR_SIZE/2 * SCALE))
            for sx_w, sy_w, v in self.last_ir[2]:
                sp = w2c(sx_w, sy_w)
                c  = IR_ON if v else IR_OFF
                pygame.draw.circle(self.canvas_surf, c, sp, ir_rad)
                pygame.draw.circle(self.canvas_surf, (0,0,0), sp, ir_rad, 1)

        # Mode badge (top-left of canvas)
        mode_col = {'IR': ACCENT, 'CAM': PURPLE, 'HYB': TEAL}[self.mode]
        mode_lbl = {'IR': 'IR Sensors', 'CAM': 'Camera', 'HYB': 'Hybrid'}[self.mode]
        badge_w  = 90
        pygame.draw.rect(self.canvas_surf, (*mode_col, 50),
                         (8, 8, badge_w, 20), border_radius=4)
        pygame.draw.rect(self.canvas_surf, mode_col, (8, 8, badge_w, 20), 1, border_radius=4)
        bl = self.fn_xs.render(mode_lbl, True, mode_col)
        self.canvas_surf.blit(bl, (8 + badge_w//2 - bl.get_width()//2, 13))

        # Run/Pause badge
        rc = GREEN_OK if self.running else (100, 100, 110)
        rl_txt = "RUNNING" if self.running else "PAUSED"
        rw2 = 70
        pygame.draw.rect(self.canvas_surf, rc,
                         (CANVAS_W - rw2 - 8, 8, rw2, 20), 1, border_radius=4)
        rb = self.fn_xs.render(rl_txt, True, rc)
        self.canvas_surf.blit(rb, (CANVAS_W - rw2//2 - 8 - rb.get_width()//2, 13))

        # Speed bar
        spd  = abs(self.stats['spd'])
        barW = int(min(1.0, spd/(self.sl_spd.value*2+0.01)) * 100)
        pygame.draw.rect(self.canvas_surf, (40, 50, 60),
                         (8, CANVAS_H-22, 100, 10), border_radius=3)
        if barW > 0:
            pygame.draw.rect(self.canvas_surf, ACCENT,
                             (8, CANVAS_H-22, barW, 10), border_radius=3)
        sv = self.fn_xs.render(f"{spd:.2f} m/s", True, TEXT_SEC)
        self.canvas_surf.blit(sv, (8, CANVAS_H-12))

        # Sim speed badge
        sm = self.fn_xs.render(f"SIM {self.sim_mult}x", True, TEXT_SEC)
        self.canvas_surf.blit(sm, (CANVAS_W-50, CANVAS_H-12))

    def draw_panel(self):
        p = self.panel_surf
        p.fill(PANEL_BG)

        # Title
        ttl = self.fn_ttl.render("Fil d'Ariane Sim", True, TEXT_PRI)
        p.blit(ttl, (PANEL_W//2 - ttl.get_width()//2, 10))
        pygame.draw.line(p, PANEL_LINE, (8, 28), (PANEL_W-8, 28))

        # Mode buttons
        for b in self.mode_buttons:
            b.draw(p, self.fn_sm)

        # Section: Robot params
        sec = self.fn_xs.render("ROBOT & SENSOR", True, TEXT_SEC)
        p.blit(sec, (12, 74))
        pygame.draw.line(p, PANEL_LINE, (8, 83), (PANEL_W-8, 83))
        for sl in self.sliders:
            sl.draw(p, self.fn_sm, self.fn_xs)

        # Section: Controls
        sec2 = self.fn_xs.render("CONTROLS", True, TEXT_SEC)
        p.blit(sec2, (12, 270))
        pygame.draw.line(p, PANEL_LINE, (8, 279), (PANEL_W-8, 279))
        for b in [self.btn_play, self.btn_reset,
                  self.btn_new, self.btn_hard, self.btn_autop, self.btn_noise]:
            b.draw(p, self.fn_sm)

        # Section: Stats
        sy = 396
        sec3 = self.fn_xs.render("LIVE STATS", True, TEXT_SEC)
        p.blit(sec3, (12, sy))
        pygame.draw.line(p, PANEL_LINE, (8, sy+9), (PANEL_W-8, sy+9))

        cw2, gap = 96, 6
        s = self.stats
        spd_val = f"{abs(s['spd']):.2f}"
        err_val = f"{s['err']*100:.1f}"
        lt_val  = f"{s['lap_t']:.2f}"
        bt_val  = f"{s['best']:.2f}" if s['best'] else "--"
        lp_val  = str(s['laps'])
        pn_val  = str(s['pen'])
        st_val  = s.get('state', 'FOLLOWING')

        y0 = sy + 14
        draw_stat(p, self.fn_lg, self.fn_xs, "speed m/s", spd_val, 12,       y0, cw2, 42)
        draw_stat(p, self.fn_lg, self.fn_xs, "error cm",  err_val, 12+cw2+gap, y0, cw2, 42)
        y1 = y0 + 48
        draw_stat(p, self.fn_lg, self.fn_xs, "lap time s",lt_val,  12,       y1, cw2, 42)
        bt_c = GREEN_OK if s['best'] else TEXT_PRI
        draw_stat(p, self.fn_lg, self.fn_xs, "best lap s",bt_val,  12+cw2+gap,y1, cw2, 42, bt_c)
        y2 = y1 + 48
        draw_stat(p, self.fn_lg, self.fn_xs, "laps done", lp_val,  12,       y2, cw2, 42)
        draw_stat(p, self.fn_lg, self.fn_xs, "penalties", pn_val,  12+cw2+gap,y2, cw2, 42, RED_ERR)
        y3 = y2 + 48
        st_c = { 'FOLLOWING': GREEN_OK, 'GAP_NAV': ORANGE, 
                 'SHARP_DECISION': PURPLE, 'SHARP_BRAKE': RED_ERR, 
                 'SHARP_PIVOT': ORANGE, 'SHARP_DAMP': TEAL,
                 'BRAKING': ACCENT, 'LOST': RED_ERR }.get(st_val, TEXT_PRI)
        draw_stat(p, self.fn_lg, self.fn_xs, "robot state", st_val, 12, y3, PANEL_W-24, 42, st_c)
        
        # Error graph
        gy = y3 + 52
        sec4 = self.fn_xs.render("ERROR GRAPH", True, TEXT_SEC)
        p.blit(sec4, (12, gy - 10))
        self.err_graph.draw(p, 10, gy, PANEL_W-20, 68, self.fn_xs)

        # Keyboard hints
        hints = [
            "SPACE play/pause   R reset",
            "N / H   new track / hard track",
            "1=IR  2=Camera  3=Hybrid",
        ]
        hy = SIM_H - 50
        pygame.draw.line(p, PANEL_LINE, (8, hy-6), (PANEL_W-8, hy-6))
        for i, h in enumerate(hints):
            ht = self.fn_xs.render(h, True, TEXT_SEC)
            p.blit(ht, (PANEL_W//2 - ht.get_width()//2, hy + i*13))

    # ── Event handling ────────────────────────────────────────────────────────

    def handle_events(self):
        panel_ox = CANVAS_W   # panel is to the right of canvas
        panel_oy = 0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    return False
                if k == pygame.K_SPACE:
                    self.toggle_play()
                if k == pygame.K_r:
                    self.reset_robot()
                if k == pygame.K_n:
                    self.new_track(hard_mode=False)
                if k == pygame.K_h:
                    self.new_track(hard_mode=True)
                if k == pygame.K_1:
                    self._set_mode('IR')
                if k == pygame.K_2:
                    self._set_mode('CAM')
                if k == pygame.K_3:
                    self._set_mode('HYB')
                if k in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    self.sim_mult = {1:3, 3:8, 8:1}[self.sim_mult]
                if k in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    self.sim_mult = {1:8, 3:1, 8:3}[self.sim_mult]

            # Slider drag (panel space)
            for sl in self.sliders:
                sl.handle_event(event, panel_ox, panel_oy)

            # Button clicks
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                pos = event.pos
                if self.btn_play.hit(pos, panel_ox, panel_oy):
                    self.toggle_play()
                if self.btn_reset.hit(pos, panel_ox, panel_oy):
                    self.reset_robot()
                if self.btn_new.hit(pos, panel_ox, panel_oy):
                    self.new_track(hard_mode=False)
                if self.btn_hard.hit(pos, panel_ox, panel_oy):
                    self.new_track(hard_mode=True)
                if self.btn_autop.hit(pos, panel_ox, panel_oy):
                    self._auto_pid()
                if self.btn_noise.hit(pos, panel_ox, panel_oy):
                    self.noise = not self.noise
                    self.btn_noise.label  = f"Noise: {'ON' if self.noise else 'OFF'}"
                    self.btn_noise.active = self.noise
                for b in self.mode_buttons:
                    if b.hit(pos, panel_ox, panel_oy):
                        self._set_mode(b.label[:3].upper().strip())

        return True

    def toggle_play(self):
        self.running = not self.running
        self.btn_play.label  = "Pause" if self.running else "Play"
        self.btn_play.active = self.running

    def _set_mode(self, m: str):
        # normalise button label → mode key
        if 'IR'  in m.upper(): m = 'IR'
        if 'CAM' in m.upper(): m = 'CAM'
        if 'HYB' in m.upper(): m = 'HYB'
        self.mode = m
        for b in self.mode_buttons:
            b.active = (b.label[:3].upper().strip()[:3] == m[:3])
        self.robot.pid.reset()

    def _auto_pid(self):
        spd = self.sl_spd.value
        kp  = round(3.0 + spd*4.0, 1)
        kd  = round(0.3 + spd*0.6, 2)
        self.sl_kp.value = min(kp, self.sl_kp.max_val)
        self.sl_kd.value = min(kd, self.sl_kd.max_val)
        self.sl_ki.value = 0.0
        self.robot.pid.reset()

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self):
        while True:
            dt_real = self.clock.tick(60) / 1000.0

            if not self.handle_events():
                break

            if self.running:
                self.accum += dt_real * self.sim_mult
                steps = 0
                while self.accum >= DT and steps < 60:
                    self.sim_step()
                    self.accum -= DT
                    steps      += 1

            self.draw_canvas()
            self.draw_panel()

            self.screen.blit(self.canvas_surf, (0, 0))
            self.screen.blit(self.panel_surf,  (CANVAS_W, 0))

            # Separator
            pygame.draw.line(self.screen, PANEL_LINE,
                             (CANVAS_W, 0), (CANVAS_W, SIM_H), 1)

            pygame.display.flip()

        pygame.quit()
        sys.exit()


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)
    Sim().run()

