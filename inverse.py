import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.fft import fft, ifft

from kpz import kpz


def err(a, x, t, h, ha):
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
    ha : Callable
        Function which takes the parameters and returns the model heights at the given times based on the parameters of a and the initial heights h0

    Returns
    -------
    float
        Mean squared error between the data and the model
    """
    h_a = ha(a, x, t, h[0])
    return np.average((h - h_a)**2)


def ha(a, x, t, h0):
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
    def kpz_wrapper(t, y):
        return kpz(y, t, a[0], 1, a[1], a[2])
    h_a = solve_ivp(kpz_wrapper, t, h0, vectorized=True)
    return np.array([h_a.y[:, 0], h_a.y[:, -1]])


def inverse(h, t, n=100):
    """Inverse method for the Kardar-Parisi-Zhang equation

    Parameters
    ----------
    h : NDArray
        Array of initial and final heights [h_0 h_final]. Must be of shape (2, len(x))
    t : NDArray
        Array of initial and final times [t_0 t_final]
    n : int, optional
        The number of fourier components to keep, by default 100

    Returns
    -------
    scipy.optimize.OptimizeResult
        The result of the optimization from scipy.optimize.minimize
    """
    firsth, lasth = h
    
    # smooth data by course graining fourier components as in Campos 2013
    firsth_smooth = smooth(firsth, n)
    lasth_smooth = smooth(lasth, n)
    firstErr = np.average((firsth - firsth_smooth)**2)
    lastErr = np.average((lasth - lasth_smooth)**2)
    # firstErr, lastErr # error from fourier course graining
    
    # initialize parameters and normalize data
    # initialize parameters
    a = np.array([1, 1, 1]) # c v lambda
    h = np.array([firsth_smooth, lasth_smooth])

    # normalize the data
    x = np.linspace(0, 1, len(firsth_smooth))
    y_bounds = np.array([h.min(), h.max()])
    h = normalize(h, y_bounds)
    # t = np.array([0, 1], float)
    # todo: normalize the time. requires modification of other parts of the code

    # plt.clf()
    # plt.plot(x, h[0], label='first fireline')
    # plt.plot(x, h[1], label='last fireline')
    # plt.legend()
    
    ha_result = minimize(err, a, args=(x, t, h, ha))
    return h, x, ha_result, y_bounds


def smooth(data, n=100):
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


def normalize(data, bounds):
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


def truncate(data, bounds):
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