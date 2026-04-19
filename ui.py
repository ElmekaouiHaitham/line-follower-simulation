import pygame
from typing import List
from config import *

# ─── Slider UI widget ─────────────────────────────────────────────────────────

class Slider:
    def __init__(self, label: str, min_val: float, max_val: float, value: float,
                 fmt: str = ".2f", x: int = 0, y: int = 0, w: int = 140, h: int = 4):
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = value
        self.fmt = fmt
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.dragging = False

    def draw(self, surf: pygame.Surface, font_sm, font_xs):
        # Label
        lbl = font_xs.render(self.label, True, TEXT_SEC)
        surf.blit(lbl, (self.x, self.y - 13))
        # Track
        pygame.draw.rect(surf, PANEL_LINE, (self.x, self.y, self.w, self.h), border_radius=2)
        # Fill
        fill_w = int((self.value - self.min_val) / (self.max_val - self.min_val) * self.w)
        if fill_w > 0:
            pygame.draw.rect(surf, ACCENT,
                             (self.x, self.y, fill_w, self.h), border_radius=2)
        # Thumb
        tx = self.x + fill_w
        pygame.draw.circle(surf, ACCENT, (tx, self.y + self.h//2), 7)
        pygame.draw.circle(surf, TEXT_PRI, (tx, self.y + self.h//2), 4)
        # Value
        val_txt = font_sm.render(format(self.value, self.fmt), True, TEXT_PRI)
        surf.blit(val_txt, (self.x + self.w + 6, self.y - 5))

    def handle_event(self, event, offset_x: int, offset_y: int) -> bool:
        mx, my = pygame.mouse.get_pos()
        lx, ly = mx - offset_x, my - offset_y
        hit = (self.x - 10 <= lx <= self.x + self.w + 10 and
               self.y - 10 <= ly <= self.y + self.h + 10)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hit:
            self.dragging = True
        if event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        if self.dragging:
            t   = max(0.0, min(1.0, (lx - self.x) / self.w))
            raw = self.min_val + t * (self.max_val - self.min_val)
            # snap to 2 decimals
            step = 10 ** (-int(self.fmt[-1]) if self.fmt[-1].isdigit() else -2)
            self.value = round(raw / step) * step
            return True
        return False

# ─── Stat card ───────────────────────────────────────────────────────────────

def draw_stat(surf, font_lg, font_xs, label, value, x, y, w=95, h=44, color=TEXT_PRI):
    pygame.draw.rect(surf, (38, 38, 44), (x, y, w, h), border_radius=6)
    pygame.draw.rect(surf, PANEL_LINE,   (x, y, w, h), 1, border_radius=6)
    val_s = font_lg.render(str(value), True, color)
    lbl_s = font_xs.render(label,      True, TEXT_SEC)
    surf.blit(val_s, (x + w//2 - val_s.get_width()//2, y + 6))
    surf.blit(lbl_s, (x + w//2 - lbl_s.get_width()//2, y + h - 14))

# ─── Error graph ──────────────────────────────────────────────────────────────

class ErrGraph:
    def __init__(self, max_pts: int = 300):
        self.buf  : List[float] = []
        self.max_pts = max_pts

    def push(self, e: float):
        self.buf.append(e * 100)   # convert to cm
        # We don't pop here anymore to keep full lap data if needed, 
        # but we should still limit it for the UI display if it gets too large.
        # Actually, for the UI we can just show the last N points.
        if len(self.buf) > 5000: # Safety limit
            self.buf.pop(0)

    def clear(self):
        self.buf = []

    def draw(self, surf: pygame.Surface, x: int, y: int, w: int, h: int,
             font_xs):
        pygame.draw.rect(surf, (28, 28, 32), (x, y, w, h), border_radius=5)
        pygame.draw.rect(surf, PANEL_LINE,   (x, y, w, h), 1, border_radius=5)
        # Zero line
        my = y + h//2
        pygame.draw.line(surf, (70, 70, 80), (x, my), (x+w, my))
        maxE = 4.0   # cm display range
        # Grid
        for v in [2.0, 4.0]:
            gy_p = int(my - v/maxE * h * 0.44)
            gy_n = int(my + v/maxE * h * 0.44)
            pygame.draw.line(surf, (50, 50, 58), (x, gy_p), (x+w, gy_p))
            pygame.draw.line(surf, (50, 50, 58), (x, gy_n), (x+w, gy_n))
        
        if len(self.buf) > 2:
            # For the UI, we only draw the last max_pts
            display_buf = self.buf[-self.max_pts:]
            pts = []
            for i, e in enumerate(display_buf):
                ex = x + int(i / self.max_pts * w)
                ey = int(my - max(-maxE, min(maxE, e)) / maxE * h * 0.44)
                pts.append((ex, ey))
            if len(pts) > 1:
                pygame.draw.lines(surf, ACCENT, False, pts, 2)
            # Current dot
            pygame.draw.circle(surf, RED_ERR, pts[-1], 4)
        lbl = font_xs.render("lateral error (cm)", True, TEXT_SEC)
        surf.blit(lbl, (x + 4, y + 4))

    def save_to_file(self, path: str, font_xs):
        """Saves the entire buffer to an image file."""
        w, h = 600, 300
        surf = pygame.Surface((w, h))
        surf.fill((22, 22, 26)) # PANEL_BG
        
        # Re-use drawing logic but for the full buffer
        x, y, gw, gh = 20, 40, w - 40, h - 80
        pygame.draw.rect(surf, (28, 28, 32), (x, y, gw, gh), border_radius=5)
        pygame.draw.rect(surf, (50, 50, 55), (x, y, gw, gh), 1, border_radius=5)
        
        my = y + gh // 2
        pygame.draw.line(surf, (70, 70, 80), (x, my), (x + gw, my))
        maxE = 4.0
        for v in [2.0, 4.0]:
            gy_p = int(my - v/maxE * gh * 0.44)
            gy_n = int(my + v/maxE * gh * 0.44)
            pygame.draw.line(surf, (50, 50, 58), (x, gy_p), (x+gw, gy_p))
            pygame.draw.line(surf, (50, 50, 58), (x, gy_n), (x+gw, gy_n))

        if len(self.buf) > 1:
            n = len(self.buf)
            pts = []
            for i, e in enumerate(self.buf):
                ex = x + int(i / n * gw)
                ey = int(my - max(-maxE, min(maxE, e)) / maxE * gh * 0.44)
                pts.append((ex, ey))
            pygame.draw.lines(surf, (66, 165, 245), False, pts, 2) # ACCENT
            
        title = font_xs.render(f"Lap Error Graph - {len(self.buf)} samples", True, (230, 230, 230))
        surf.blit(title, (x, 10))
        
        pygame.image.save(surf, path)


# ─── Button ──────────────────────────────────────────────────────────────────

class Button:
    def __init__(self, label: str, x: int, y: int, w: int, h: int,
                 color=ACCENT):
        self.label = label
        self.rect  = pygame.Rect(x, y, w, h)
        self.color = color
        self.active = False

    def draw(self, surf: pygame.Surface, font_sm):
        bg = self.color if self.active else (45, 45, 52)
        bd = self.color
        pygame.draw.rect(surf, bg, self.rect, border_radius=6)
        pygame.draw.rect(surf, bd, self.rect, 1, border_radius=6)
        txt = font_sm.render(self.label, True,
                             (20, 20, 20) if self.active else TEXT_PRI)
        surf.blit(txt, (self.rect.x + self.rect.w//2 - txt.get_width()//2,
                        self.rect.y + self.rect.h//2 - txt.get_height()//2))

    def hit(self, pos, off_x=0, off_y=0) -> bool:
        return self.rect.move(off_x, off_y).collidepoint(pos)
