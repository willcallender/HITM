# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
import pickle
import scipy.fftpack as fft

# %%
# numerical solver for the Kardar-Parisi-Zhang equation
def kpz(h, t, c, dx, v, lamb, y=1, n=0):
    Gamma = np.empty_like(h)
    # second-order central
    Gamma[1:-1] = h[2:] + h[:-2] - 2*h[1:-1]
    # second order forward
    Gamma[0] = h[2] + h[0] - 2*h[1]
    # second order backward
    Gamma[-1] = h[-1] + h[-3] - 2*h[-2]
    
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
    # todo psi is blowing up on real data, could be error in implementation
    # could also be that "fourier truncation" is needed... whatever that means
    
    if n:
        # gaussian noise
        noise = np.random.normal(0, n, h.shape)
    else:
        noise = np.zeros_like(h)
        
    retval = c + 1/(dx**2) * (v*Gamma + Psi*lamb*0.5) + noise
    return retval

# %%
def eq11(a, H, n, x, t):
    # eq 11 in Campos 2013
    # Gamma is del2 h
    # Psi is (del h)^2
    c, v, lamb = a
    lamb /= 2
    _, d2h, dh2 = H
    # truncate the fourier components??

# %%
def err(a, x, t, h, ha):
    # eq 13 in Campos 2013
    h_a = ha(a, x, t, h[0])
    return np.average((h - h_a)**2)

# %%
# define constants
c = 0.02
v = 0.1
lamb = 0.3
D = 4.5e-6

# %%
# define init conds
timestamps = np.linspace(0, 10)
x = np.linspace(0, 1)
# y0 = -(np.linspace(-1, 1)**2) + 1 # creates inverted parabola
# y0 = np.sin(2*np.pi*x) # creates sine wave
y0 = np.sin(2*np.pi*x) + x # creates sine wave with linear slope

# %%
plt.plot(x, y0)

# %%
plt.clf()

# %%
# function callable by scipy
def f(t, y):
    return kpz(y, t, c, 1, v, lamb, n=0)

# %%
# solve over time
h_result = solve_ivp(f, [0, 1], y0, vectorized=True)

# %%
front1 = h_result.y[:, 0]
front2 = h_result.y[:, -1]

# %%
# init all the things
a = np.array([1, 1, 1])
x = np.linspace(0, 1)
t = np.array([h_result.t[0], h_result.t[-1]])
h = np.array([front1, front2])

# %%
# minimize the error
def ha(a, x, t, h0):
    def func(t, y):
        return kpz(y, t, a[0], 1, a[1], a[2])
    h_a = solve_ivp(func, t, h0, vectorized=True)
    return np.array([h_a.y[:, 0], h_a.y[:, -1]])


# %%
ha_result = minimize(err, a, args=(x, t, h, ha))
ha_result

# %%
# reconstructed kpz eqn
a_opt = ha_result.x
final = solve_ivp(lambda t, y: kpz(y, t, a_opt[0], 1, a_opt[1], a_opt[2]), [0, 1], y0, vectorized=True)

# %%
# compare original and reconstructed
plt.plot(front2, 'b-', label='original')
plt.plot(final.y[:, -1], 'r.', label='reconstructed')
plt.legend()

# %%
# load real data
filename = 'radialSpread.pickle'
with open(filename, 'rb') as f:
    firespread = pickle.load(f)

# %%
t = np.array([10, 20])
firstFireline = firespread[t[0]]
firstx = firstFireline[0]
firsth = firstFireline[1]
sortIdx = np.argsort(firstx)
firstx = firstx[sortIdx]
firsth = firsth[sortIdx]
plt.plot(firstx, firsth, label='first fireline')

lastFireline = firespread[t[1]]
lastx = lastFireline[0]
lasth = lastFireline[1]
sortIdx = np.argsort(lastx)
lastx = lastx[sortIdx]
lasth = lasth[sortIdx]
plt.plot(lastx, lasth, label='last fireline')
plt.legend()

# %%
# coarse-grain the data by truncating the Fourier components


a = np.array([1, 1, 1])
x = np.linspace(-np.pi, np.pi, 1024)
h = np.zeros((2, len(x)))
h[0] = np.interp(x, firstx, firsth)
h[1] = np.interp(x, lastx, lasth)
plt.clf()
plt.plot(x, h[0], label='first fireline')
plt.plot(x, h[1], label='last fireline')

# %%
ha_result = minimize(err, a, args=(x, t, h, ha))
ha_result

# %%
a_opt = ha_result.x
final = solve_ivp(lambda t, y: kpz(y, t, a_opt[0], 1, a_opt[1], a_opt[2]), t, h[0], vectorized=True)
plt.clf()
plt.plot(x, h[1], 'b-', label='original')
plt.plot(x, final.y[:, -1], 'r.', label='reconstructed')


