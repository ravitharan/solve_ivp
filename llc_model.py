import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton

def OneClick(event):
    if event.button is MouseButton.RIGHT:
        print(f'{round(event.xdata)}{round(event.ydata)}')

C = 1.13e-6
Lr = 2.25e-6
Lm = 11.93e-6
R = 50
VDC = 30
RESONANT_FREQ = 1 / (2 * np.pi * np.sqrt(Lr * C))

FREQ = -100
if len(sys.argv) >= 2:
    FREQ = float(sys.argv[1])

if len(sys.argv) >= 3:
    R = float(sys.argv[2])

s_freq = RESONANT_FREQ + FREQ
s_period = 1 / s_freq
resolution = s_period / 100

g_t_switch_l = s_period / 2
g_t_switch_h = s_period

g_T = []
g_V = []
g_last_t = 0
g_last_v = 0
def inv_volt(t):
    global g_t_switch_l, g_t_switch_h
    if (t < g_t_switch_l):
        V = VDC
    elif (t < g_t_switch_h):
        V = -VDC
    else:
        g_t_switch_l += s_period
        g_t_switch_h += s_period
        V = VDC
    return V
    
def llc_model(t, y):
    '''
    dq_dt = i
    di_dt = (v - R * (i - i1) - q/C) / Lr
    di1_dt = R * (i - i1) / Lm
    '''
    global g_T, g_V, g_last_t, g_last_v
    q = y[0]
    i = y[1]
    i1 = y[2]
    dq_dt = i
    V = inv_volt(t)
    di_dt = (V - R * (i - i1) - q / C) / Lr
    di1_dt = R * (i - i1) / Lm

    if V != g_last_v:
        if t > g_last_t:
            g_T.append(t - resolution)
            g_V.append(g_last_v)
            g_T.append(t)
            g_V.append(V)
            g_last_t = t
            g_last_v = V
    return [dq_dt, di_dt, di1_dt]

def cal_rms(T, Y, period):
    t1 = T[-1] - 8 * period
    inx = np.searchsorted(T, t1)
    Y_slice = Y[inx:]
    return np.sqrt(np.mean(np.square(Y_slice)))

t_interval = (0, 0.01)
y0 = np.array([0, 0, 0])
V_FUN = inv_volt
g_T = []
g_V = []
sol = solve_ivp(llc_model,
                t_interval,
                y0,
                max_step=resolution)

vout = R * (sol.y[1] - sol.y[2])
rms = cal_rms(sol.t, vout, s_period)
print(f'freq {s_freq:.2f}, vout {rms:.2f}')

fig, axe = plt.subplots()
fig.canvas.mpl_connect('button_press_event', OneClick)

axe.plot(g_T, g_V, label=f'vin')
axe.plot(sol.t, sol.y[1], label=f'i')
axe.plot(sol.t, vout, label=f'vout')
axe.set_title(f'LLC response')
axe.legend(loc='upper right')
axe.grid(True)
plt.show()
