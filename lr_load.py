import sys
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import inverter
import cmath
import argparse

VDC = 350
V_RMS = 230
V_AMPL = V_RMS * np.sqrt(2)
FREQ = 50
ω = 2 * np.pi * FREQ
PWM_FREQ = 5e3
Z_MAG = 150
T_SIMULATION = 2.0
OPT_TOL = 1e-3

def OneClick(event):
    if event.button is MouseButton.RIGHT:
        print(f'{round(event.xdata)}{round(event.ydata)}')

def ac_volt(t, i, v_ampl, freq):
    ω = 2 * np.pi * freq
    v_ac = v_ampl * np.sin(ω * t)
    return v_ac

def lr_model(t, y, V_FUN, v_ampl, freq, L, R):
    '''
    di_dt = (dv_dt - (R * i)) / L
    '''
    i = y[0]
    V = V_FUN(t, i, v_ampl, freq)
    di_dt = (V - (R * i)) / L
    return [di_dt]

def cal_rms(T, I, t1, t2):
    i1 = np.searchsorted(T, t1)
    i2 = np.searchsorted(T, t2)
    I_slice = I[i1:i2]
    return np.mean(I_slice), np.sqrt(np.mean(np.square(I_slice)))

def rms_dt_adjust(dt_adjust):

    inverter.g_dt_adjust = dt_adjust

    t_interval = (0, T_SIMULATION)
    y = np.array([0, 0])

    V_FUN = inverter.inv_volt
    inverter.g_t_next_pwm = 0
    sol = solve_ivp(lr_model,
                    t_interval,
                    y,
                    max_step=1e-5,
                    args=(V_FUN, V_AMPL, FREQ, L, R))

    avg, rms = cal_rms(sol.t, sol.y[0], sol.t[-1] - 8 / FREQ, sol.t[-1])
    return rms

def optimize_dta(phase, dt):
    Z = cmath.rect(Z_MAG, phase * np.pi/180)
    R = Z.real
    L = Z.imag / ω
    EXP_RMS = V_RMS / abs(Z)

    inverter.g_vdc = VDC
    inverter.g_pwm_freq = PWM_FREQ
    inverter.g_pwm_period = 1 / inverter.g_pwm_freq
    inverter.g_dead_time = dt

    dta_1 = -0.5
    dta_2 = 0.5
    loop = 0

    err_1 = EXP_RMS - rms_dt_adjust(dta_1)
    err_2 = EXP_RMS - rms_dt_adjust(dta_2)

    while loop < 25:
        dta = (dta_1 + dta_2) / 2
        err = EXP_RMS - rms_dt_adjust(dta)

        if abs(err) <= OPT_TOL:
            break
        loop += 1
        if np.sign(err_1) == np.sign(err):
            dta_1 = dta
            err_1 = err
        elif np.sign(err_2) == np.sign(err):
            dta_2 = dta
            err_2 = err
    return dta, err, loop

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p",
            "--phase_angle",
            type=float,
            default=45.0,
            help="LR impedence phase angle in degree")
    parser.add_argument("-d",
            "--dead_time",
            type=float,
            default=20.0,
            help="Dead time percentage with pwm_period")
    parser.add_argument("-a",
            "--dt_adjust",
            type=float,
            default=0,
            help="Dead time adjust within [-0.5, 0.5]")
    parser.add_argument("-c",
            "--compare",
            action='store_true',
            default=True,
            help="Compare inverter responce with ideal AC source")
    parser.add_argument("-o",
            "--optimize",
            action='store_true',
            help="Calculate optimal dt_adjust for this condition")
    return parser.parse_args()


if __name__ == "__main__":
    args =  parse_argument()
    DEAD_TIME = args.dead_time / PWM_FREQ / 100
    Z = cmath.rect(Z_MAG, args.phase_angle * np.pi/180)
    R = Z.real
    L = Z.imag / ω
    EXP_RMS = V_RMS / abs(Z)

    inverter.g_vdc = VDC
    inverter.g_pwm_freq = PWM_FREQ
    inverter.g_pwm_period = 1 / inverter.g_pwm_freq
    inverter.g_dead_time = DEAD_TIME

    if args.optimize:
        dta, err, loop = optimize_dta(args.phase_angle, DEAD_TIME)
        print(f'Phase {args.phase_angle}, DT {args.dead_time}, DTA {dta:.4f}, err {err:.3f}, loop {loop}')

    elif args.compare:
        print(f'phase {args.phase_angle}, DT {DEAD_TIME} ({args.dead_time}%), DT adjust {args.dt_adjust}')

        t_interval = (0, T_SIMULATION)
        y = np.array([0, 0])

        V_FUN = inverter.inv_volt
        inverter.g_dt_adjust = args.dt_adjust
        inverter.g_t_next_pwm = 0
        sol = solve_ivp(lr_model,
                        t_interval,
                        y,
                        max_step=1e-5,
                        args=(V_FUN, V_AMPL, FREQ, L, R))

        avg, rms = cal_rms(sol.t, sol.y[0], sol.t[-1] - 8 / FREQ, sol.t[-1])
        print(f'inv current: DTA {args.dt_adjust}, mean {avg:.3f}, rms {rms:.3f}, rms error {rms - EXP_RMS:.3f}')

        optimal_dt_adjust = 0.5 * np.cos(args.phase_angle * np.pi / 180)
        inverter.g_dt_adjust = optimal_dt_adjust
        inverter.g_t_next_pwm = 0
        opt_sol = solve_ivp(lr_model,
                        t_interval,
                        y,
                        max_step=1e-5,
                        args=(V_FUN, V_AMPL, FREQ, L, R))

        opt_avg, opt_rms = cal_rms(opt_sol.t, opt_sol.y[0], opt_sol.t[-1] - 8 / FREQ, opt_sol.t[-1])
        print(f'inv current: DTA {optimal_dt_adjust:.3f}, mean {opt_avg:.3f}, rms {opt_rms:.3f}, rms error {opt_rms - EXP_RMS:.3f}')

        V_FUN = ac_volt
        inverter.g_t_next_pwm = 0
        ac_sol = solve_ivp(lr_model,
                           t_interval,
                           y,
                           max_step=1e-5,
                           args=(V_FUN, V_AMPL, FREQ, L, R))
        ac_avg, ac_rms = cal_rms(ac_sol.t, ac_sol.y[0], ac_sol.t[-1] - 8 / FREQ, ac_sol.t[-1])
        print(f' ac current: mean {ac_avg:.3f}, rms {ac_rms:.3f}, rms error {ac_rms - EXP_RMS:.3f}')

        fig, axe = plt.subplots()
        fig.canvas.mpl_connect('button_press_event', OneClick)

        axe.plot(sol.t, sol.y[0], label=f'inverter current DTA {args.dt_adjust}')
        axe.plot(opt_sol.t, opt_sol.y[0], label=f'inverter current DTA {optimal_dt_adjust:.3f}')
        axe.plot(ac_sol.t, ac_sol.y[0], label='ac current')


        axe.set_title(f'Dead time {DEAD_TIME} ({100 * DEAD_TIME * PWM_FREQ:.2f}%), DT adjust {args.dt_adjust}, rms error {100 * abs(rms - ac_rms) / ac_rms:.2f}%')

        axe.legend(loc='upper right')
        axe.grid(True)
        plt.show()
