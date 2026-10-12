import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import sol_ivp
import sys

def OneClick(event):
    if event.button is MouseButton.RIGHT:
        print(f'{round(event.xdata)}{round(event.ydata)}')


Vd = 0.7
Rd = 0.01
Cr = 1.13e-6
Lr = 2.25e-6
RESONANT_FREQ = 1 / (2 * np.pi * np.sqrt(Lr * Cr))
Lm = 11.93e-6
C0 = 1e-5
R0 = 50
VDC = 40

INX_QR = 0
INX_I  = 1
INX_Q0 = 2
INX_I1 = 3

FREQ = -RESONANT_FREQ / 2
if len(sys.argv) >= 2:
    FREQ = float(sys.argv[1])

if len(sys.argv) >= 3:
    R0 = float(sys.argv[2])

s_freq = RESONANT_FREQ + FREQ
s_period = 1 / s_freq

def diode_forward_model(t, y, v, vm):
    '''
    states: qr, i, q0, i1
    '''
    di_dt = (v - y[INX_QR] / Cr - vm) / Lr
    di1_dt = di_dt - y[INX_I1] / C0 / R0 - vm / Lm
    dq0_dt = y[INX_I1]
    dqr_dt = y[INX_I]
    return [dqr_dt, di_dt, dq0_dt, di1_dt]


def diode_reverse_model(t, y, v, vm):
    '''
    states: qr, i, q0
    '''
    di_dt = (v - y[INX_QR] / Cr) / (Lr + Lm)
    dq0_dt = -y[INX_Q0] / C0 / R0
    dqr_dt = y[INX_I]
    return [dqr_dt, di_dt, dq0_dt]

def run_solver(t0, tf, y0):

    v_sign = 1
    v = v_sign * VDC
    t_switch = s_period / 2

    ts = [t0]
    ys = [y0]

    t = t0
    y = y0
    yr = y[:-1]
    sol_ivp.step.h_abs = 1e-4
    sol_ivp.max_step = 1e-2
    sol_ivp.max_diff = 1e-2

    diode_neg = True
    while t < tf:
        sol_ivp.step.tf = t_switch
        while t < t_switch:
            if not diode_neg:
                vm = Vd + (y[INX_I1] + y[INX_Q0] / C0 / R0) * Rd
                diff = diode_forward_model(t, y, v, vm)
                vm_left = v - y[INX_QR] / Cr - Lr * diff[INX_I]
                reversed = abs(vm_left) <= y[INX_Q0] / C0
                sol_ivp.args = (v, vm)
                t, y = sol_ivp.step(diode_forward_model, t, y)
                yr = y[:-1]
            else:
                vm = 0
                vm_left = (v - y[INX_QR] / Cr) * Lm / (Lr + Lm)
                reversed = abs(vm_left) <= y[INX_Q0] / C0
                sol_ivp.args = (v, vm)
                t, yr = sol_ivp.step(diode_reverse_model, t, yr)
                y[:3] = yr
                y[-1] = -yr[INX_Q0] / C0 / R0
            if diode_neg != reversed:
                diode_neg = reversed
            ts.append(t)
            ys.append(y)
        v_sign *= -1
        v = v_sign * VDC
        t_switch += s_period / 2

    return ts, ys

t, y = run_solver(0, 2e-4, np.zeros(4))
t = np.asarray(t)
y = np.vstack(y).T
vout = -y[INX_Q0] / C0 / R0

fig, axe = plt.subplots()
fig.canvas.mpl_connect('button_press_event', OneClick)

axe.plot(t, vout, label=f'vout')
axe.set_title(f'half rectifier')
axe.legend(loc='upper right')
axe.grid(True)
plt.show()
