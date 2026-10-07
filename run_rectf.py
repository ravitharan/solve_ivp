import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import sol_ivp

def OneClick(event):
    if event.button is MouseButton.RIGHT:
        print(f'{round(event.xdata)}{round(event.ydata)}')


C = 1e-5
R = 1000
V = 10
Vd = 0.7
Rd = 0.01
ω = 2 * np.pi * 50
HALF_RECT = 0

def is_diode_reversed(t, y):
    v = V * np.sin(ω * t)
    if HALF_RECT:
        return (v < y[0] / C + Vd)
    else:
        return (abs(v) < y[0] / C + Vd)

def diode_forward_model(t, y):
    if HALF_RECT:
        v = V * np.sin(ω * t)
    else:
        v = abs(V * np.sin(ω * t))
    i1 = (v - Vd - y[0] / C) / Rd
    i2 = y[0] / C / R
    i = i1 - i2
    dq_dt = i
    return [dq_dt]

def diode_reverse_model(t, y):
    i2 = y[0] / C / R
    i = - i2
    dq_dt = i
    return [dq_dt]

def run_solver(t0, tf, y0):

    ts = [t0]
    ys = [y0]

    t = t0
    y = y0
    sol_ivp.step.tf = tf
    sol_ivp.step.h_abs = 1e-6
    sol_ivp.max_step = 1e-6
    sol_ivp.max_diff = 1e-8

    last_diode = False
    while t < tf:
        diode = is_diode_reversed(t, y)
        if diode:
            t, y = sol_ivp.step(diode_reverse_model, t, y)
        else:
            t, y = sol_ivp.step(diode_forward_model, t, y)
        if last_diode != diode:
            print(f'{t:.3f} {diode}')
            last_diode = diode
        ts.append(t)
        ys.append(y)

    return ts, ys

t, y = run_solver(0, 0.16, np.asarray([0 * C]))
t = np.asarray(t)
v_ac = V * np.sin(ω * t)
y = np.vstack(y).T
vout = y[0] / C

fig, axe = plt.subplots()
fig.canvas.mpl_connect('button_press_event', OneClick)

axe.plot(t, vout, label=f'vout')
axe.plot(t, v_ac, label=f'v_ac')
axe.set_title(f'half rectifier')
axe.legend(loc='upper right')
axe.grid(True)
plt.show()
