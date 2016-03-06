"""
Script to re-create Figure 1 from Moreau and Sontag, 2003.

The system is repeatedly simulated in the periods between impulsive inputs,
since the inputs are assumed to bring the system exactly to the desired state
instantaneously.
"""

import systems

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# length of simulation in seconds
T = 20
# sampling time in seconds
dt = 0.001
# low target firing rate in Hz
x_low = 20
# high target firing rate in Hz
x_high = 60
# period of input in seconds
T_input = 1

# model parameters
mu0 = 200
eps = 0.01
a = 1
b = 0.01
c = 42

# initial conditions
x_i = 0
mu_i = mu0 - 1


def main():
    s_arr = np.zeros((T/dt, 3))
    mu = mu_i
    # set the state via input and simulate the time until the next input
    for i, ti in enumerate(range(0, T, T_input)):
        t = np.arange(ti, ti+T_input, dt)
        ind = range(i*len(t), (i+1)*len(t))

        # alternate between x_low and x_high
        xdes = x_low if i % 2 == 0 else x_high
        s_i = [xdes, mu]
        s = odeint(systems.integrator_eqns, s_i, t, args=(mu0, eps, a, b, c))

        s_arr[ind, 0] = t
        s_arr[ind, 1:] = s
        mu = s[-1, 1]

    plt.subplot(211)
    plt.plot(s_arr[:, 0], s_arr[:, 1], linewidth=3)
    plt.xticks(range(0, T+1, 2))
    plt.xlabel('time $t$ (s)')
    plt.yticks([0, 50, 100])
    plt.ylim(0, 100)
    plt.ylabel('$x$ (Hz)')
    plt.grid()

    plt.subplot(212)
    plt.plot(s_arr[:, 0], s_arr[:, 2]-mu0, linewidth=3)
    plt.xticks(range(0, T+1, 2))
    plt.xlabel('time $t$ (s)')
    plt.yticks([-1, 0, 1])
    plt.ylim(-1, 1)
    plt.ylabel('$\mu - \mu_0$ (Hz)')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
