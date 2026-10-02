#! /usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import math
import sys

g_t_next_pwm = 0
g_t_switch_h = 0
g_t_switch_l = 0
g_inv_sign = 1

g_t = []
g_v = []

g_vdc = 350
g_pwm_freq = 5e3
g_pwm_period = 1 / g_pwm_freq
g_dead_time = 0
g_dt_adjust = 0 # -0.5 <= x <= 0.5

def pwm_control(t, i, v_ampl, freq):
    ω = 2 * np.pi * freq
    period = 1 / freq
    t_pwm = math.floor(t * g_pwm_freq) * g_pwm_period
    t_next_pwm = math.ceil(t * g_pwm_freq) * g_pwm_period

    t_cycle = t - math.floor(t * freq) * period
    pwm_index = math.floor(t_cycle * g_pwm_freq)

    v_ac = v_ampl * np.sin(ω * (pwm_index + 0.5) * g_pwm_period)
    inv_sign = np.sign(v_ac)
    d = np.abs(v_ac) / g_vdc
    t_duty = d * g_pwm_period

    # Default switching times
    t_switch_h = t_pwm + t_duty - g_dead_time / 2
    t_switch_l = t_pwm + t_duty + g_dead_time / 2

    # Apply dead time compensation
    t_switch_h += g_dt_adjust * g_dead_time
    t_switch_l += g_dt_adjust * g_dead_time

    if t_switch_h < t_pwm:
        t_switch_h = t_pwm

    if t_switch_l > t_next_pwm:
        t_switch_l = t_next_pwm
    g_t.append(t)
    g_v.append((t_switch_h - t_pwm) * g_pwm_freq * inv_sign * g_vdc/v_ampl)

    return t_next_pwm, t_switch_h, t_switch_l, inv_sign


def inv_output(t, i, t_switch_h, t_switch_l, inv_sign):
    v_factor = 0
    if (t < t_switch_h):
        v_factor = inv_sign
    elif (t < t_switch_l):  # dead time
        if ((i < 0) and (inv_sign == 1)) or ((i > 0) and (inv_sign == -1)):
            v_factor = inv_sign
    return v_factor


def inv_volt(t, i, v_ampl, freq):
    global g_t_next_pwm, g_t_switch_h, g_t_switch_l, g_inv_sign
    if (t >= g_t_next_pwm):
        g_t_next_pwm, g_t_switch_h, g_t_switch_l, g_inv_sign = pwm_control(
            t, i, v_ampl, freq)
    return inv_output(t, i, g_t_switch_h, g_t_switch_l, g_inv_sign) * g_vdc


def onclick(event):
    if event.button is MouseButton.RIGHT:
        print(f'{round(event.xdata)} {round(event.ydata)}')


if __name__ == '__main__':

    g_vdc = 350
    V_RMS = 230
    V_AMPL = V_RMS * np.sqrt(2)
    FREQ = 50
    g_pwm_freq = 5e3
    g_pwm_period = 1 / g_pwm_freq
    g_dead_time = g_pwm_period / 10
    ω = 2 * np.pi * FREQ

    if len(sys.argv) == 2:
        g_dead_time = float(sys.argv[1])

    T = np.arange(0, 4 / FREQ, g_pwm_period / 1000)
    V_inv = np.zeros(len(T))
    V = V_AMPL * np.sin(ω * T)

    for i in range(len(T)):
        V_inv[i] = inv_volt(T[i], 0, V_AMPL, FREQ)

    fig, axe = plt.subplots()
    fig.canvas.mpl_connect('button_press_event', onclick)

    axe.plot(T, V, label="ac voltage", marker='')
    axe.plot(T, V_inv, label="inv voltage", marker='')
    axe.legend(loc='upper right')
    axe.set_title('')
    axe.grid(True)
    plt.show()
