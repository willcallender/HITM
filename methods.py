import numpy as np
import numpy.typing as npt
from math import ceil
from typing import Callable, Tuple, Any, Union, Literal
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, kron, spmatrix
from scipy.sparse.linalg import spsolve

def Screened_Poisson_2D(lamb: float, N: int, f: Union[Callable[[np.float_, np.float_], np.float_], npt.ArrayLike]) -> Tuple[spmatrix, npt.NDArray, npt.NDArray]:
    """Implements a finite difference method to solve the 2D 
    screened Poisson equation

    Parameters
    ----------
    lamb : float
        The parameter lambda
    N : int
        The number of discretization points
    f : (np.float_, np.float_) -> np.float_ | npt.ArrayLike
        Either a function handle or a matrix specifying the RHS of the ODE

    Returns
    -------
    Tuple[spmatrix, NDArray, NDArray]
        A tuple of U, x, y. U is the approximate solution in the domain, x 
        specifies the x coordinates of each gridpoint, and y is the same as x 
        but for the y coordinates.
    """
    # Initialize solver
    h = 1 / (N + 1)
    A = eye(N, N)
    B = -Finite_Difference_Sparse(N, h)
    L = lamb * eye(N**2, N**2) - (kron(A, B) + kron(B, A))
    [x, y] = np.meshgrid(np.arange(h, 1, h), np.arange(h, 1, h))
    
    # Handle f being either a function handle or a vector
    if callable(f):
        x = x.flatten()
        y = y.flatten()
        F = f(x, y)
        x = x.reshape(N, N)
        y = y.reshape(N, N)
    else:
        F = np.array(f)
        F = F.flatten()
    
    U = spsolve(L, F).reshape(N, N)
   
    return U, x, y

def Poisson_Solver(f: Callable[[npt.ArrayLike], npt.ArrayLike], limits: npt.ArrayLike, bounds: npt.ArrayLike, N: int, matType: Literal['dense', 'sparse']='sparse') -> Tuple[npt.NDArray, npt.NDArray]:
    # initialize parameters
    a, b = limits
    dx = (b-a)/(N+1)
    
    # get finite difference matrix
    if matType == 'dense':
        A: npt.NDArray = Finite_Difference_Dense(N, dx)
    else:
        A = Finite_Difference_Sparse(N, dx).tocsc()
    
    x = np.linspace(a, b, N+2)[1:-1]
    F = f(x)
    F[0] += bounds[0] / (dx**2)
    F[-1] += bounds[-1] / (dx**2)
    if matType == 'dense':
        U: npt.NDArray = np.linalg.solve(A, F)
    else:
        U: npt.NDArray = spsolve(A, F)
    return U, x

def Finite_Difference_Dense(N: int, dx: float):
    A = np.eye(N,N,0)*2 - np.eye(N,N,1) - np.eye(N,N,-1)
    A /= dx**2
    return A

def Finite_Difference_Sparse(N: int, dx: float):
    A = diags([[-1]*(N-1), [2]*(N), [-1]*(N-1)], [-1, 0, 1])
    A /= dx**2
    return A

def milstein(a: Callable[[np.float_, np.float_], np.float_], \
    b: Callable[[np.float_, np.float_], np.float_], \
    by: Callable[[np.float_, np.float_], np.float_], y0: float, T: float, \
    dt: float) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Implements the Milstein method for solving a SDE

    Parameters
    ----------
    a : (y: float_, t: float) -> float_
        Drift coefficient, as a function of y(t) and t.
    b : (y: float, t: float) -> float_
        Diffusion coefficient, as a function of y(t) and t.
    by : (y: float, t: float) -> float_
        Derivative of diffusion coefficient, as a function of y(t) and t.
    y0 : float
        The initial value.
    T : float
        The time horizon.
    dt : float
        The time step.

    Returns
    -------
    Tuple[NDArray[float_], NDArray[float_]]
        Returns a tuple of two 1D arrays. The first is all the time values of t 
        and the second is all the values of Y.
    """
    # Initialization
    N = ceil(T/dt) # calculate the number of steps
    t = np.linspace(0, T, N+1)
    Y = np.zeros_like(t)
    Y[0] = y0
    
    # define expressions for newton's method
    def g(z, t, Yn, dW):
        return z - 0.5 * dt * a(z, t) - Yn - 0.5 * dt * a(Yn, t) + \
            b(Yn, t) * dW + 0.5 * b (Yn, t) * by(Yn, t) * (dW**2 - dt)
    
    def gp(z, t, Yn, dW):
        return 1 # this can't be right but maybe Newton's will work regardless?
    
    # Main time-stepping loop
    for n in range(N):
        dW = np.random.normal(0, np.sqrt(dt))
        Y[n+1], iter = Newtons_Method(lambda z: g(z, t[n], Y[n], dW), \
            lambda z: gp(z, t[n], Y[n], dW), Y[n])
    
    return t, Y

BDF_Beta = np.array([1, 2/3, 6/11, 12/25, 60/137, 60/147])

BDF_Alpha = np.array(
    [[1],
     [4/3, -1/3],
     [18/11, -9/11, 2/11],
     [48/25, -36/25, 16/25, -3/25],
     [300/137, -300/137, 200/137, -75/137, 12/137],
     [360/147, -450/147, 400/147, -225/147, 72/147, -10/147]], object
)

def BDF(m: int, y0: npt.NDArray[np.float_], T: float, dt: float, f: Callable[[float, float], float], fp: Callable[[float, float], float]):
    """Implements the Backwards Difference Formula for solving IVPs

    m : int
        The number of steps to consider for every new timestep.
    y0 : npt.NDArray[np.float_]
        The initial values at times {0, dt, ... , (m-1)dt}.
    T : float
        The time horizon.
    dt : float
        The time step.
    f : Callable[[float, float], float]
        A function handle for the ODE. f(Y, t). Returns a float.
    fp : Callable[[float, float], float]
        A function handle for the partial derivative of f with respect to y.

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Returns (t, y), a tuple of two 1D arrays. The first is all the time
        values of t and the second is all the values of Y.
    """
    # Initialization
    N = ceil(T/dt) # calculate the number of steps
    t = np.linspace(0, T, N+1)
    Y = np.zeros_like(t)
    Y[:m] = y0
    y = lambda t: 1/(1+np.exp(-t))
    y_t = y(t)
    
    a = BDF_Alpha[m-1][::-1]
    B = BDF_Beta[m-1]
    
    # define implicit function and its derivative
    def g(z, s, tn1):
        return z - s - B * dt * f(z, tn1)
    def gp(z, tn1):
        return 1 - B * dt * fp(z, tn1)
    
    # Main time-stepping loop
    for n in range(m-1, N):
        # apply BDF
        # summation component of rhs of BDF
        s = np.sum(a[:m] * Y[n-m+1:n+1])
        # implicitly solve
        Y[n+1], iter = Newtons_Method(lambda z: g(z, s, t[n+1]),\
            lambda z: gp(z, t[n+1]), Y[n])
        # print(Y[n+1], y_t[n+1])
    return t, Y

# define the b coefficients for Adams-Bashforth
Adams_Bashforth_Table = np.array(
    [[1],
    [3/2, -1/2],
    [23/12, -4/3, 5/12],
    [55/24, -59/24, 37/24, -3/8],
    [1901/720, -1387/360, 109/30, -637/360, 251/720]], object)

def Adams_Bashforth(m: int, y0: float, T: float, dt: float, f: Callable[[float, float], float]):
    """Implements the Adams-Bashforth multistep method for solving IVPs

    m : int
        The number of steps to consider for every new timestep.
    y0 : float
        The initial value.
    T : float
        The time horizon.
    dt : float
        The time step.
    f : Callable[[float, float], float]
        A function handle for the ODE. f(Y, t). Returns a float.

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Returns (t, y), a tuple of two 1D arrays. The first is all the time
        values of t and the second is all the values of Y.
    """
    # Initialization
    N = ceil(T/dt) # calculate the number of steps
    t = np.linspace(0, T, N+1)
    Y = np.zeros_like(t)
    Y[0] = y0
    
    # calculate missing timesteps using RK4
    _, Y[:m] = RK4(y0, t[m-1], dt, f)
    b = Adams_Bashforth_Table[m-1]
    
    # Main time-stepping loop
    for n in range(m-1, N):
        # apply general Adams-Bashforth method
        Ys = Y[n-m+1:n+1][::-1] # avoid confusing python with negative indices
        ts = t[n-m+1:n+1][::-1]
        Y[n+1] = Y[n] + dt * np.sum(b * f(Ys, ts))
    return t, Y

def RK4(y0: float, T: float, dt: float, f: Callable[[float, float], float]):
    """Implements a basic 4th stage RK method to solve an IVP

    y0 : float
        The initial value.
    T : float
        The time horizon.
    dt : float
        The time step.
    f : Callable[[float, float], float]
        A function handle for the ODE. f(Y, t). Returns a float.

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Returns (t, y), a tuple of two 1D arrays. The first is all the time
        values of t and the second is all the values of Y.
    """
    A = np.array([[0,   0,   0,   0  ],
                  [1/2, 0,   0,   0  ],
                  [0,   1/2, 0,   0  ],
                  [0,   0,   1,   0, ]])
    c = np.array([0, 1/2, 1/2, 1])
    b = np.array([1/6, 1/3, 1/3, 1/6])
    return RKmethod(A, c, b, y0, T, dt, f)

def RKmethod(A: npt.NDArray[np.float_], c: npt.NDArray[np.float_],\
    b: npt.NDArray[np.float_], y0: float, T: float, dt: float,\
        f: Callable[[float, float], float]):
    """Implements a general Runge-Kutta method to solve an IVP

    Parameters
    ----------
    A : NDArray[float]
        The matrix A that defines a valid RK method.
    c: NDArray[float]
        The vector c that defines a valid RK method.
    b: NDArray[float]
        The vector b that defines a valid RK method.
    y0 : float
        The initial value.
    T : float
        The time horizon.
    dt : float
        The time step.
    f : Callable[[float, float], float]
        A function handle for the ODE. f(Y, t). Returns a float.

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Returns (t, y), a tuple of two 1D arrays. The first is all the time
        values of t and the second is all the values of Y.
    """
    # Initialization
    N = ceil(T/dt) # calculate the number of steps
    t = np.linspace(0, T, N+1)
    Y = np.zeros_like(t)
    Y[0] = y0
    s = b.size
    k = np.zeros((s, 1))
    b = b.reshape((1, s))

    # Main time-stepping loop
    for n in range(N):
        # apply general RK method
        for j in range(s):
            k[j] = dt*f(Y[n] + A[j, :] @ k, t[n] + c[j] * dt)
        Y[n+1] = Y[n] + b @ k;
    return t, Y

def error(expected, observed, relative=True):
    err = observed - expected
    if relative: err = abs(err / expected)
    return err

def Midpoint_Method(y0: float, T: float, dt: float, f: Callable[[float, float],\
    float]) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Uses the midpoint method to solve an IVP

    Parameters
    ----------
    y0 : float
        The initial value.
    T : float
        The time horizon.
    dt : float
        The time step.
    f : Callable[[float, float], float]
        A function handle for the ODE. f(Y, t). Returns a float.

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Returns a tuple of two 1D arrays. The first is all the time values of t 
        and the second is all the values of Y.
    """
    # Initialization
    N = ceil(T/dt) # calculate the number of steps
    t = np.linspace(0, T, N+1)
    Y = np.zeros_like(t)
    Y[0] = y0

    # Main time-stepping loop
    for n in range(N):
        # apply midpoint method
        k1 = dt*f(Y[n], t[n])
        k2 = dt*f(Y[n] + k1/2, t[n] + dt/2)
        Y[n+1] = Y[n] + k2
    return t, Y


def stability_plot(amp: Callable[[npt.NDArray[np.complex_]], npt.NDArray[np.complex_]], limits: Tuple[float,\
    float, float, float], N: int=1000):
    """Generates a plot of the region of stablity for a given amplification 
    factor. Alters the current figure to allow for modification of plotting 
    elements before and after this function is called (ie title, show, etc).

    Parameters
    ----------
    amp : Callable[[complex], complex]
        A function handle for the amplification factor Λ(z).
    limits : Tuple[float, float, float, float]
        The bounds of the plot. (xmin, xmax, ymin, ymax).
    N : int, optional
        The number of discretization points in each direction, by default 1000.
    """
    xmin, xmax, ymin, ymax = limits
    dx = (xmax-xmin)/(N-1)
    dy = (ymax-ymin)/(N-1)
    [x, y] = np.meshgrid(np.linspace(xmin, xmax, N), np.linspace(ymin, ymax, N))
    z = x + 1j * y
    StabRegion = amp(z)
    StabRegion[np.abs(StabRegion) >= 1] = 1
    StabRegion[np.abs(StabRegion) < 1] = 0
    plt.imshow(np.real(StabRegion), extent=limits, interpolation="none",\
        cmap='plasma')
    plt.xlabel('Blue is stable, yellow is unstable')

def Forward_Euler(y0: float, T: float, dt: float, f: Callable[[float, float],\
    float]) -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Uses the Forward Euler method to solve an IVP.

    Parameters
    ----------
    y0 : float
        The initial value.
    T : float
        The time horizon.
    dt : float
        The time step.
    f : Callable[[float, float], float]
        A function handle for the ODE. f(Y, t). Returns a float.

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Returns a tuple of two 1D arrays. The first is all the time values of t 
        and the second is all the values of Y.
    """
    # Initialization
    N = ceil(T/dt) # calculate the number of steps
    t = np.linspace(0, T, N+1)
    Y = np.zeros_like(t)
    Y[0] = y0

    # Main time-stepping loop
    for n in range(N):
        # apply forward euler method
        Y[n+1] = Y[n] + dt*f(Y[n], t[n])
    return t, Y

def Backward_Euler(y0: float, T: float, dt: float, f: Callable[[float, float],\
    float], fp: Callable[[float, float], float])\
        -> Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]:
    """Uses the Backward Euler method to solve an IVP using Newtons_Method.

    Parameters
    ----------
    y0 : float
        The initial value.
    T : float
        The time horizon.
    dt : float
        The time step.
    f : Callable[[float, float], float]
        A function handle for the ODE. f(Y, t). Returns a float.
    fp : Callable[[float, float], float]
        A function handle for the derivative of the ODE. f'(Y, t). Returns a 
        float.

    Returns
    -------
    Tuple[npt.NDArray[np.float_], npt.NDArray[np.float_]]
        Returns a tuple of two 1D arrays. The first is all the time values of t 
        and the second is all the values of Y.
    """
    # Initialization
    N = ceil(T/dt)
    t = np.linspace(0, T, N+1)
    Y = np.zeros_like(t)
    Y[0] = y0
    n = 0
    
    def g(z, Yn, tn1):
        return z - Yn - dt * f(z, tn1)
    def gp(z, tn1):
        return 1 - dt * fp(z, tn1)

    while t[n] < T:
        Y[n+1]=Newtons_Method(lambda z: g(z, Y[n], t[n+1]),\
            lambda z: gp(z, t[n+1]), Y[n], 10, 10e-10)[0]
        n += 1
    
    return t, Y

def Newtons_Method(f: Callable[[float], float], fp: Callable[[float], float],\
    x0: float, M: int=100, e: float=10e-10) -> Tuple[float, int]:
    """Implements Newton's Method for finding roots.

    Parameters
    ----------
    f : callable[[float], float]
        The function of interest. Accepts x as a float and returns f(x) as a
        float.
    fp : callable[[float], float]
        The first derivative of the function of interest. Accepts x as a float
        and returns f'(x) as a float.
    x0 : float
        The initial guess.
    M : int
        The maximum number of iterations to perform. Must be a finite, positive
        integer.
    e : float
        The maximum absolute difference between f(x*) and 0.


    Returns
    -------
    Tuple[float, int]
        (x*, iter) where x* is the approximate root and iter is the number of
        iterations it took to find. If iter=M it likely implies Newton's method
        did not converge.
    """

    x = x0 # initialize the approximation, x means x*
    k = 0 # keep track of the number of iterations    

    # newton's method
    # print(f'{x:.2e}')    
    while abs(f(x)) > e and k < M:
        x = x - f(x)/fp(x)
        k += 1
        # print(f'{x:.2e}')    
    # return the root and the number of iterations
    return x, k
