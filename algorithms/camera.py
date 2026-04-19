import math

def read_camera(r, grid, look: float):
    """Returns (error, lost, detected_world_points)."""
    fx =  math.cos(r.theta)
    fy =  math.sin(r.theta)
    px = -math.sin(r.theta)
    py =  math.cos(r.theta)
    sw  = 0.20
    pts = []
    for di in range(1, 9):
        d = look * di / 8
        for si in range(16):
            lo  = -sw + 2*sw*si/15
            qx  = r.x + fx*d + px*lo
            qy  = r.y + fy*d + py*lo
            if grid.on_line(qx, qy):
                pts.append((qx, qy, lo, d))
    if not pts:
        return r.last_cam_err, True, []
    ws = sum(p[2] * (p[3]/look) for p in pts)
    wt = sum(p[3]/look for p in pts)
    err = ws / wt
    r.last_cam_err = err
    return err, False, pts
