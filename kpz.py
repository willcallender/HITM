import numpy as np
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult
from numba import vectorize, jit, njit, guvectorize
import numpy.typing as npt
from numba import float64, int64


@guvectorize([(float64[:], float64, float64, float64, float64, float64[:])], '(n),(),(),(),()->(n)', nopython=True)
def kpz(h: npt.NDArray[np.floating], c: float, dx: float, v: float, lamb: float, ret: npt.NDArray[np.floating]):
    """Numerical solver for the Kardar-Parisi-Zhang equation

    Parameters
    ----------
    h : NDArray
        Initial height
    c : float
        The constant c in the Kardar-Parisi-Zhang equation
    dx : float
        The spacing between points in the x direction
    v : float
        The constant v in the Kardar-Parisi-Zhang equation
    lamb : float
        The constant lambda in the Kardar-Parisi-Zhang equation
    ret : NDArray
        The array to store the result in

    Returns
    -------
    NDArray
        The height at the next time step
    """
    y = 1
    n = 0
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
    
    # exponential distribution
    noise = np.random.exponential(1, size=h.shape)
    
    # normal distribution
    # noise = np.random.normal(0, 1, size=h.shape)
    
    # absolute normal
    # noise = np.abs(np.random.normal(0, 1, size=h.shape))
        
    ret[:] = c + 1/(dx**2) * (v*Gamma + Psi*lamb*0.5) + noise
    # return ret

# def solveKPZ(h, t, a):

def solveKPZ(h: npt.NDArray[np.floating], t: npt.NDArray[np.integer], a: npt.NDArray[np.floating]) -> OdeResult:
    """Find the height of the fireline at each time step based on the parameters in a

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
        ret = np.empty_like(y)
        kpz(y, a[0], 1, a[1], a[2], ret)
        return ret
    
    timespan = (t[0], t[-1])
    # t = t[1:-1]
    # if len(t) == 0:
    #     t = None
    # solve the ODE at all time steps
    solution = solve_ivp(kpz_wrapper, timespan, h, t_eval=t) #, vectorized=True)
    return solution


# def biharmonic():
#     """Numerical solver for the Kardar-Parisi-Zhang equation with biharmonic term

#     Parameters
#     ----------
#     h : NDArray
#         Initial height
#     c : float
#         The constant c in the Kardar-Parisi-Zhang equation
#     dx : float
#         The spacing between points in the x direction
#     v : float
#         The constant v in the Kardar-Parisi-Zhang equation
#     lamb : float
#         The constant lambda in the Kardar-Parisi-Zhang equation
#     ret : NDArray
#         The array to store the result in

#     Returns
#     -------
#     NDArray
#         The height at the next time step
#     """