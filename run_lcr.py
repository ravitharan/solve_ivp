import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import sol_ivp

def OneClick(event):
    if event.button is MouseButton.RIGHT:
        print(f'{round(event.xdata)}{round(event.ydata)}')


C = 0.94e-6
L = 2.2e-6
R = 1
VDC = 30
def lcr_model(t, y):
    '''
    dq_dt = i
    di_dt = (v - R * i - q/C) / L
    '''
    q = y[0]
    i = y[1]
    dq_dt = i
    di_dt = (VDC - R * i - q / C) / L

    return [dq_dt, di_dt]

def run_solver(t0, tf, y0):

    ts = [t0]
    ys = [y0]

    t = t0
    y = y0
    sol_ivp.step.tf = tf
    sol_ivp.step.h_abs = 1e-6
    sol_ivp.max_diff = 1e-3
    sol_ivp.max_step = 1e-4

    while t < tf:
        t, y = sol_ivp.step(lcr_model, t, y)
        ts.append(t)
        ys.append(y)

    return ts, ys

t, y = run_solver(0, 1e-4, np.asarray([0, 0]))
t = np.asarray(t)
y = np.vstack(y).T

fig, axe = plt.subplots()
fig.canvas.mpl_connect('button_press_event', OneClick)

axe.plot(t, y[1], label=f'LCR current')
axe.set_title(f'LR response')
axe.legend(loc='upper right')
axe.grid(True)
plt.show()
