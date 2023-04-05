import numpy as np
from scipy.integrate import solve_ivp


# numerical solver for the Kardar-Parisi-Zhang equation
def kpz(h, t, c, dx, v, lamb, y=1, n=0):
    """Numerical solver for the Kardar-Parisi-Zhang equation

    Parameters
    ----------
    h : NDArray
        Initial height
    t : NDArray
        Array of initial and final times [t_0 t_final]
    c : float
        The constant c in the Kardar-Parisi-Zhang equation
    dx : float
        The spacing between points in the x direction
    v : float
        The constant v in the Kardar-Parisi-Zhang equation
    lamb : float
        The constant lambda in the Kardar-Parisi-Zhang equation
    y : int, optional
        Parameter used in the discretization of the non-linear term, by default 1
    n : int, optional
        The standard deviation of the Gaussian term, by default 0

    Returns
    -------
    NDArray
        The height at the next time step
    """
    Gamma = np.empty_like(h)
    # second-order central
    Gamma[1:-1] = h[2:] + h[:-2] - 2*h[1:-1]
    # second order forward
    Gamma[0] = h[2] + h[0] - 2*h[1]
    # second order backward
    Gamma[-1] = h[-1] + h[-3] - 2*h[-2]
    # todo: how did they handle boundary terms in the paper?
    
    # nonlinear term
    Psi = np.empty_like(h)
    # inside terms
    Psi[1:-1] = 1/(2*y+1)*(
    (h[2:]-h[1:-1])**2 + \
    2*y*(h[2:]-h[1:-1])*(h[1:-1]-h[:-2]) + \
    (h[1:-1]-h[:-2])**2
    )
    # solution for boundary terms not explained in paper, check other papers ref [24-30]
    Psi[0] = Psi[1]
    Psi[-1] = Psi[-2]
    
    if n:
        # gaussian noise
        noise = np.random.normal(0, n, h.shape)
    else:
        noise = np.zeros_like(h)
        
    retval = c + 1/(dx**2) * (v*Gamma + Psi*lamb*0.5) + noise
    return retval


def solveKPZ(h, t, a):
    """Find the height of the fireline at each time step based on the parameters a

    Parameters
    ----------
    h : NDArray
        Array of initial height. Must be of shape (len(x))
    t : NDArray
        Time points of interest
    a : NDArray
        Array of parameters [c v lambda]

    Returns
    -------
    scipy.integrate._ivp.ivp.OdeResult
        The result of the ODE solver from scipy.integrate.solve_ivp
    """
    def kpz_wrapper(t, y):
        return kpz(y, t, a[0], 1, a[1], a[2])
    
    timespan = (t[0], t[-1])
    # solve the ODE at all time steps
    solution = solve_ivp(kpz_wrapper, timespan, h, t_eval=t, vectorized=True)
    return solution


