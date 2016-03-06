import numpy as np


def integrator_eqns(s, t, mu0, eps, a, b, c):
    """
    System of first-order differential equations for simulating the neural
    integrator. The states are [x, mu], where x is the neuron firing rate and
    mu is the tuning parameter.

    Parameters
    ----------
    s : ndarray, shape (num_states,)
        Value of each state variable at the current time.
    t : float
        Current time (s).
    mu0 : float
        Natural neural firing decay rate (s^-1).
    eps : float
        Overall tuning coefficient.
    a : float
        Portion of tuning rate due to current firing rate.
    b : float
        Portion of tuning rate due to current tuning parameter.
    c : float
        Offset applied to tuning rate.

    Returns
    -------
    d_s : ndarray, shape (num_states,)
        First derivative of each state variable at the current time.
    """
    x, mu = s

    d_x = (mu - mu0)*x
    d_mu = eps * (-a*x - b*mu + c)

    return d_x, d_mu


def oscillator_eqns(s, t, mu0, lam, om):
    """
    System of first-order differential equations for simulating the nonlinear
    oscillator. The states are [x, d_x, mu], where x is the position, d_x is
    the velocity, and mu is the tuning parameter.

    Parameters
    ----------
    s : ndarray, shape (num_states,)
        Value of each state variable at the current time.
    t: float
        Current time (s).
    mu0 : float
        Natural decay rate, damping coefficient applied to velocity.
    lam : float
        Damping coefficient applied to cubed velocity.
    om : float
        Natural frequency.

    Returns
    -------
    d_s : ndarray, shape (num_states,)
        First derivative of each state variable at the current time.
    """
    x, d_x, mu = s

    r = np.sqrt(x**2 + (d_x/om)**2)
    dd_x = -(mu0 - mu)*d_x - lam*d_x**3 - om**2*x
    d_mu = 1/(1 + r**2) - mu + 0.5

    return d_x, dd_x, d_mu
