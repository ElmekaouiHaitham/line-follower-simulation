# Le Fil d'Ariane — Line Follower Robot Simulator

A 2-D physics simulator for an autonomous line-following robot, built with Python and Pygame.
The robot navigates a procedurally generated track using a PID controller fed by configurable sensor modes.

---

## Features

- **PID line-following** with real-time tunable Kp, Ki, Kd sliders
- **Gap logic** — robot coasts straight when the line is temporarily lost, stops after a configurable timeout
- **3 sensor modes** switchable at runtime:
  | Key | Mode | Description |
  |-----|------|-------------|
  | `1` | IR | 12-element infrared sensor array |
  | `2` | Camera | Simulated camera-based centroid detection |
  | `3` | Hybrid | IR + camera fusion |
- **Procedural track generation** — random tracks and harder challenge tracks (`N` / `H`)
- **First-order motor lag** simulation (GA12-N20 model)
- **Live stats panel** — speed, error, state, lap time, sensor readings

---

## Project Structure

```
projet inp/
├── main.py          # Entry point — simulation loop, event handling, sliders
├── robot.py         # Robot model, PID controller, gap logic
├── track.py         # Procedural track generation
├── ui.py            # Pygame UI, panel rendering
├── config.py        # All constants (world size, robot dimensions, colours…)
└── algorithms/
    ├── ir.py        # IR sensor array simulation
    ├── camera.py    # Camera-based line detection
    └── hybrid.py    # Sensor fusion (IR + Camera)
```

---

## Requirements

- Python 3.10+
- `pygame`
- `numpy`

Install dependencies:
```bash
pip install pygame numpy
```

---

## Running

```bash
python main.py
```

### Controls

| Key | Action |
|-----|--------|
| `SPACE` | Play / Pause |
| `R` | Reset robot (same track) |
| `N` | Generate new random track |
| `H` | Generate hard track |
| `1` / `2` / `3` | Switch sensor mode (IR / Camera / Hybrid) |
| `+` / `-` | Simulation speed (1× / 3× / 8×) |
| Mouse | Drag sliders in the right panel |
| `ESC` / `Q` | Quit |

---

## Configuration

All physical and visual constants live in `config.py`:

| Constant | Default | Description |
|----------|---------|-------------|
| `N_IR` | 12 | Number of IR sensors |
| `IR_SPACING` | 9 mm | Center-to-center sensor gap |
| `WHEEL_BASE` | 13 cm | Distance between wheels |
| `DT` | 1/200 s | Simulation timestep |
| `WW × WH` | 2.1 × 1.4 m | World dimensions |

PID gains and the gap timeout (`GAP_TIMEOUT = 0.5 s`) can be changed in `robot.py` or adjusted live via the sliders.

---

## License

MIT
