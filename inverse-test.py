import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize
from scipy.fft import fft, ifft
import pickle

from kpz import *
from inverse import *

x_bounds = np.array([100, 250])
t_bounds = np.array([23, 43])
# t_bounds = np.array([10, 13])

filename = 'spread.pkl'
with open(filename, 'rb') as f:
    spread = pickle.load(f)
spread = np.array(spread)
y_bounds = np.array([0, spread.max()])

h0, h1, verifyh, ha_result, y_bounds, finals, errs = runExperiment(spread, t_bounds[0], t_bounds[1], np.array([40], int), x_bounds, y_bounds)
print(ha_result)

x = np.arange(len(h0)).astype(float)
plt.plot(x, h0, label='h0')
plt.plot(x, h1, '--', label='h1')
plt.plot(x, verifyh[0], label='original')
plt.plot(x, finals[0].y[:, -1], label='reconstructed')
# plt.ylim(0, 1)
plt.legend()
plt.show()
