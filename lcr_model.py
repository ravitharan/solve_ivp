import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

def OneClick(event):
    if event.button is MouseButton.RIGHT:
        print(f'{round(event.xdata)}{round(event.ydata)}')

C = 0.94e-6
L = 2.2e-6
R = 1
VDC = 30
RESONANT_FREQ = 1 / (2 * np.pi * np.sqrt(L * C))
FREQ = -100
s_freq = RESONANT_FREQ + FREQ
s_period = 1 / s_freq

g_t_switch_l = s_period / 2
g_t_switch_h = s_period

def inv_volt(t):
    global g_t_switch_l, g_t_switch_h
    if (t < g_t_switch_l):
        return VDC
    elif (t < g_t_switch_h):
        return -VDC
    else:
        g_t_switch_l += s_period
        g_t_switch_h += s_period
        return VDC
    
AC_FREQ = 100
ω = 2 * np.pi * AC_FREQ

def ac_volt(t):
    return VDC * np.sin(ω * t)
    
def lcr_model(t, y, V_FUN):
    '''
    dq_dt = i
    di_dt = (v - R * i - q/C) / L
    '''
    q = y[0]
    i = y[1]
    dq_dt = i
    di_dt = (V_FUN(t) - R * i - q / C) / L

    return [dq_dt, di_dt]

t_interval = (0, 0.01)
y0 = np.array([0, 0])
V_FUN = inv_volt
sol = solve_ivp(lcr_model,
                t_interval,
                y0,
                max_step=s_period / 100,
                args=(V_FUN,))

fig, axe = plt.subplots()
fig.canvas.mpl_connect('button_press_event', OneClick)

axe.plot(sol.t, sol.y[1], label=f'LCR current')
axe.set_title(f'LCR response')
axe.legend(loc='upper right')
axe.grid(True)
plt.show()
