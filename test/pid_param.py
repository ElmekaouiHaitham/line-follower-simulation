import numpy as np
import matplotlib.pyplot as plt

# ── PID parameters to tune ──────────────────────────
Kp = 30
Ki = 0.01
Kd = 0.8
BASE_SPEED = 150        # arbitrary motor units

# ── Simulate a step disturbance (robot knocked off line) ──
dt = 0.005              # 5ms loop = 200Hz (your real loop target)
T  = 3.0                # simulate 3 seconds
steps = int(T / dt)

# ── State ────────────────────────────────────────────
error     = np.zeros(steps)
integral  = 0.0
prev_err  = 0.0
position  = 5.0         # start 5 units off center

errors, positions, corrections = [], [], []

for i in range(steps):
    error = position    # error = how far from center

    # PID
    integral  = np.clip(integral + error * dt, -100, 100)  # anti-windup
    derivative = (error - prev_err) / dt
    correction = Kp * error + Ki * integral + Kd * derivative

    # Apply to simulated position (simple model)
    position  -= correction * dt * 0.8   # 0.8 = simulated robot response
    prev_err   = error

    errors.append(error)
    positions.append(position)
    corrections.append(correction)

# ── Plot ─────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7))

axes[0].plot(positions, label="Position (offset from line)", color="red")
axes[0].axhline(0, color="green", linestyle="--", label="Target (center)")
axes[0].set_title(f"PID Response  —  Kp={Kp}  Ki={Ki}  Kd={Kd}")
axes[0].set_ylabel("Offset (cm)")
axes[0].legend(); axes[0].grid(True)

axes[1].plot(corrections, label="PID correction", color="blue")
axes[1].set_ylabel("Correction"); axes[1].set_xlabel("Time steps")
axes[1].legend(); axes[1].grid(True)

plt.tight_layout()
plt.show()