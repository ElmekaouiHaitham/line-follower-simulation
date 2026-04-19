import math
import random
from config import N_IR, IR_SPACING, IR_FWDOFF, IR_SIZE

def read_ir(r, grid, noise: bool = False):
    """Returns (error, lost, sensor_positions_and_values)."""
    px = -math.sin(r.theta)
    py =  math.cos(r.theta)
    fx =  math.cos(r.theta)
    fy =  math.sin(r.theta)
    
    # Centre of the sensor array bar
    bx  = r.x + fx * IR_FWDOFF
    by  = r.y + fy * IR_FWDOFF
    
    readings = []
    # Semi-width of a single sensor for area sampling
    sw = IR_SIZE / 2.0
    
    for i in range(N_IR):
        # Center of this specific sensor
        off = (i - (N_IR-1)/2.0) * IR_SPACING
        cx, cy = bx + px*off, by + py*off
        
        # Check 3 points across the 3mm sensor width (left edge, center, right edge)
        # This simulates the physical area of the sensor detection
        v = 0
        for dx in [-sw, 0, sw]:
            if grid.on_line(cx + px*dx, cy + py*dx):
                v = 1
                break
                
        if noise and random.random() < 0.02:
            v = 1 - v
        readings.append((cx, cy, v))

    s  = sum(rd[2] for rd in readings)
    ws = sum((i - (N_IR-1)/2.0) * IR_SPACING * rd[2]
             for i, rd in enumerate(readings))
    
    if s == 0:
        err  = r.last_ir_err * 1.5 # Dampened extrapolation
        lost = True
    else:
        err  = ws / s
        lost = False
        r.last_ir_err = err
    
    return err, lost, readings
