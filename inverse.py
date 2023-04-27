import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft
from numba import njit, jit, vectorize, guvectorize
import numba.types as nbt
from numba.types import float64
import traceback

from typing import Tuple, Callable
import numpy.typing as npt


from kpz import *


"""
TODO: FINISH DOCSTRINGS AND TYPE ANNOTATIONS
"""

# def err(a, x, t, h, ha):

def err(a: npt.NDArray[np.floating], x: npt.NDArray[np.floating], t: npt.NDArray[np.integer], h: npt.NDArray[np.floating], ha: Callable[[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.integer], npt.NDArray[np.floating]], npt.NDArray[np.floating]]) -> float:
    """Calculate the error between the data and the model. This is equation 13 in Campos 2013.

    Parameters
    ----------
    a : NDArray
        Array of parameters [c v lambda]
    x : NDArray
        Array of x values
    t : NDArray
        Array of initial and final times [t_0 t_final]
    h : NDArray
        Array of initial and final heights [h_0 h_final]. Must be of shape (2, len(x))
    ha : Callable ha(a, x, t, h0)
        Function which takes the parameters and returns the model heights at the given times based on the parameters of a and the initial heights h0

    Returns
    -------
    float
        Mean squared error between the data and the model
    """
    h_a = ha(a, x, t, h[0])
    return np.average((h - h_a)**2)


# def ha(a, x, t, h0):

def ha(a: npt.NDArray[np.floating], x: npt.NDArray[np.floating], t: npt.NDArray[np.integer], h0: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Calculate the model heights at the given times based on the parameters of a and the initial heights h0

    Parameters
    ----------
    a : NDArray
        Array of parameters [c v lambda]
    x : NDArray
        Array of x values
    t : NDArray
        Array of initial and final times [t_0 t_final]
    h0 : NDArray
        Array of initial heights. Must be of shape (len(x))
        
    Returns
    -------
    NDArray
        Array of initial and final heights [h_0 h_final]. Must be of shape (2, len(x))
    """
    dx = x[1] - x[0]
    if np.any(np.diff(x) != dx):
        raise ValueError("x must be uniformly spaced")
    
    def kpz_wrapper(t: float, y: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        ret = np.zeros_like(y)
        kpz(y, a[0], dx, a[1], a[2], ret)
        return ret
    h_a = solve_ivp(kpz_wrapper, t, h0)# , vectorized=True)
    return np.array([h_a.y[:, 0], h_a.y[:, -1]])


# def inverse(h, t, n=100):

def inverse(h: npt.NDArray[np.floating], t: npt.NDArray[np.floating], y_bounds: npt.NDArray[np.floating] = None, n: int = 100) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], OptimizeResult, npt.NDArray[np.floating]]:
    """Inverse method for the Kardar-Parisi-Zhang equation

    Parameters
    ----------
    h : NDArray
        Array of initial and final heights [h_0 h_final]. Must be of shape (2, len(x))
    t : NDArray
        Array (or tuple) of initial and final times [t_0 t_final]
    y_bounds : NDArray, optional
        Array of the y bounds [y_min, y_max], by default None. If None, the bounds are calculated from the data
    n : int, optional
        The number of fourier components to keep, by default 100

    Returns
    -------
    Tuple[NDArray, NDArray, OptimizeResult, NDArray]
        h : Array of initial and final heights [h_0 h_final]. Must be of shape (2, len(x))
        x : Array of x values
        ha_result : OptimizeResult
            Result of the optimization
        y_bounds : Array of the y bounds [y_min, y_max]
    """
    firsth, lasth = h
    
    # smooth data by course graining fourier components as in Campos 2013
    firsth_smooth = smooth(firsth, n)
    lasth_smooth = smooth(lasth, n)
    # firstErr = np.average((firsth - firsth_smooth)**2)
    # lastErr = np.average((lasth - lasth_smooth)**2)
    # firstErr, lastErr # error from fourier course graining
    
    # initialize parameters and normalize data
    # initialize parameters
    a = np.array([1, 1, 1]) # c v lambda
    h = np.array([firsth_smooth, lasth_smooth])

    # normalize the data
    # x = np.linspace(0, 1, len(firsth_smooth))
    x = np.arange(0, len(firsth_smooth)).astype(float)
    if y_bounds is None:
        y_bounds = np.array([h.min(), h.max()])
    h = normalize(h, y_bounds)
    # t = np.array([0, 1], float)
    # todo: normalize the time. requires modification of other parts of the code
    
    ha_result = minimize(err, a, args=(x, t, h, ha))
    return h, x, ha_result, y_bounds


# def smooth(data, n=100):
def smooth(data: npt.NDArray[np.floating], n: int = 100) -> npt.NDArray[np.floating]:
    """Smooth data by course graining fourier components as in Campos 2013

    Parameters
    ----------
    data : NDArray
        Array of data to smooth
    n : int, optional
        The number of fourier components to keep, by default 100

    Returns
    -------
    NDArray
        Array of smoothed data
    """
    fft_data = fft(data)
    fft_data[n:] = 0
    return np.real(ifft(fft_data))


# def normalize(data, bounds):
def normalize(data: npt.NDArray[np.floating], bounds: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Normalize data according to the given bounds

    Parameters
    ----------
    data : NDArray
        Data to normalize
    bounds : NDArray
        The bounds to normalize the data to. Must be of shape (2,). The value of the first element will be mapped to 0 and the value of the second element will be mapped to 1.

    Returns
    -------
    NDArray
        Array of normalized data
    """
    data_norm = data - bounds[0]
    data_norm /= bounds[1] - bounds[0]
    return data_norm


# def truncate(data, bounds):
def truncate(data: npt.NDArray, bounds: npt.NDArray[np.integer]) -> npt.NDArray:
    """Truncate data according to the given bounds
    
    Parameters
    ----------
    data : NDArray
        Data to truncate
    bounds : NDArray
        The bounds to truncate the data to. Must be of shape (2,). The value of the first element will be the starting index and the value of the second element will be the ending index.
        
    Returns
    -------
    NDArray
        Array of truncated data
    """
    data_trunc = data[bounds[0]:bounds[1]]
    return data_trunc

# def runExperiment(spread, firstIdx, lastIdx, verifyIdx, x_bounds, n=100):

def runExperiment(spread: npt.NDArray[np.floating], firstIdx: int, lastIdx: int, verifyIdx: npt.NDArray[np.integer], x_bounds: npt.NDArray[np.integer], y_bounds: npt.NDArray[np.floating] = None, n: int = 100) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], OptimizeResult, npt.NDArray[np.floating], list, npt.NDArray[np.floating]]:
    """Fit the Kardar-Parisi-Zhang equation to the given data and compare to verification data

    Parameters
    ----------
    spread : NDArray
        Array of spread data. Must be of shape (len(t), 2, len(x)) (I think)
    firstIdx : int
        Time index of the first fireline
    lastIdx : int
        Time index of the last fireline
    verifyIdx : NDArray
        Time indices of the verification firelines
    x_bounds : NDArray
        The bounds to truncate the data to. Must be of shape (2,). The value of the first element will be the starting index and the value of the second element will be the ending index.
    y_bounds : NDArray, optional
        The bounds of the data. Has shape (2,). The value of the first element will be mapped to 0 and the value of the second element will be mapped to 1. If None, the bounds will be calculated from the data, by default None
    n : int, optional
        The number of fourier components to keep, by default 100

    Returns
    -------
    tuple
        Tuple of (h0, h1, verifyh, ha_result, y_bounds, finals, errs)

    Notes
    -----
    h0 : NDArray
        Array of initial heights. Has shape (len(x))
    h1 : NDArray
        Array of final heights. Has shape (len(x))
    verifyh : NDArray
        Array of verification heights. Has shape (len(x))
    ha_result : scipy.optimize.OptimizeResult
        The result of the optimization from scipy.optimize.minimize
    y_bounds : NDArray
        The bounds of the data. Has shape (2,). The value of the first element will be mapped to 0 and the value of the second element will be mapped to 1.
    finals : scipy.integrate.OdeResult
        The approximated fireline for each of the verification firelines
    errs : NDArray
        Array of errors. Has shape (len(verifyIdx), len(x))
    """
    firsth = spread[firstIdx][1]
    lasth = spread[lastIdx][1]
    
    firsth = truncate(firsth, x_bounds)
    lasth = truncate(lasth, x_bounds)
    
    t = np.array([firstIdx, lastIdx], float)
    h, x, ha_result, y_bounds = inverse(np.array([firsth, lasth], float), t, y_bounds, n)
    a_opt = ha_result.x
    a_opt = np.array(a_opt)
    
    finals = []
    errs = np.zeros((len(verifyIdx), len(x)))
    verifyhs = np.zeros((len(verifyIdx), len(x)))
    for i, id in enumerate(verifyIdx):
        verifyh = spread[id][1]
        verifyh = truncate(verifyh, x_bounds)
        verifyh = smooth(verifyh, n)
        verifyh = normalize(verifyh, y_bounds)
        verifyhs[i] = verifyh
        t[1] = id
        final = solveKPZ(h[0], t, a_opt)
        finals.append(final)
        # try:
        #     errs[i] = verifyh - final.y[:, -1]
        # except TypeError as e:
        #     print(errs, i, verifyh, final, type(errs), type(i), type(verifyh), type(final.y), sep='\n')
        #     print(firstIdx, lastIdx, verifyIdx, x_bounds, y_bounds, n, sep='\n')
        #     traceback.print_exc()
        #     raise e
        errs[i] = verifyh - final.y[:, -1]
    
    return h[0], h[1], verifyhs, ha_result, y_bounds, finals, errs

