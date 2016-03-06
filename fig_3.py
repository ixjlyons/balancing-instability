"""
Script to re-create Figure 3 from Moreau and Sontag, 2003.
"""

import systems

from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt


# length of simulation in seconds
T = 35
# sampling time in seconds
dt = 0.01

# model parameters
mu0 = 1
lam = 1
om = 1

# initial conditions
x_i = 0
d_x_i = 0.3
mu_i = mu0 - 10


def main():
    t = np.arange(0, T, dt)
    s_i = [x_i, d_x_i, mu_i]
    s = odeint(systems.oscillator_eqns, s_i, t, args=(mu0, lam, om))

    plt.subplot(211)
    plt.plot(t, s[:, 0], linewidth=2)
    plt.xticks(range(0, T+1, 5))
    plt.xlabel('time $t$')
    plt.ylim(-1, 1)
    plt.yticks([-1, 0, 1])
    plt.ylabel('$x$')
    plt.grid()

    plt.subplot(212)
    plt.plot(t, s[:, 2]-mu0, linewidth=2)
    plt.xlabel('time $t$')
    plt.ylim(-1, 1)
    plt.yticks([-10, 0, 10])
    plt.ylabel('$\mu - \mu_0$')
    plt.grid()

    plt.show()


if __name__ == '__main__':
    main()
