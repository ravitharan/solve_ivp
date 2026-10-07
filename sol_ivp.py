import numpy as np

'''
https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
https://en.wikipedia.org/wiki/Dormand%E2%80%93Prince_method
'''
n_stages = 6
RK_C = np.array([0, 1/5, 3/10, 4/5, 8/9, 1])
RK_A = np.array([
    [0, 0, 0, 0, 0],
    [1/5, 0, 0, 0, 0],
    [3/40, 9/40, 0, 0, 0],
    [44/45, -56/15, 32/9, 0, 0],
    [19372/6561, -25360/2187, 64448/6561, -212/729, 0],
    [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656]
])
RK_B = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84])

max_step=1e-3
max_diff = 1e-3
args = ()

def rk_step(fun, t, y, h):
    K = np.empty((n_stages, y.size), dtype=np.double)
    K[0] = fun(t, y, *args)
    for s, (a, c) in enumerate(zip(RK_A[1:], RK_C[1:]), start=1):
        dy = np.dot(K[:s].T, a[:s]) * h
        K[s] = fun(t + c * h, y + dy, *args)

    y_new = y + h * np.dot(K.T, RK_B)

    return y_new

def step(fun, t, y):

    min_step = 10 * np.abs(np.nextafter(t, np.inf) - t)

    if step.h_abs > max_step:
        step.h_abs = max_step
    elif step.h_abs < min_step:
        step.h_abs = min_step

    step_accepted = False

    while not step_accepted:
        if step.h_abs < min_step:
            raise RuntimeError('Stem size smaller than min_step') 

        h = step.h_abs
        t_new = t + h

        if t_new > step.tf:
            t_new = step.tf

        h = t_new - t
        step.h_abs = np.abs(h)

        y_new = rk_step(fun, t, y, h)
        diff = abs(y_new - y).max()

        if diff < max_diff:
            if diff < max_diff / 2:
                step.h_abs *= 2
            step_accepted = True
        else:
            step.h_abs *= 0.5

    return t_new, y_new


