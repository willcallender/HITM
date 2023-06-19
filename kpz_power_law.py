# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

from kpz import *
from inverse import *

# %%
x_bounds = np.array([100, 250])
t_bounds = np.array([10, 42])
# t_bounds = np.array([10, 13])

# %%
filename = 'spread.pkl'
with open(filename, 'rb') as f:
    spread = pickle.load(f)
spread = np.array(spread)
y_bounds = np.array([0, spread.max()])

# %%
h0, h1, verifyh, ha_result, y_bounds, finals, errs = runExperiment(spread, t_bounds[0], t_bounds[1], np.array([]), x_bounds, y_bounds)

# %%
a = ha_result.x
a

# %%
result = solveKPZ(h0, np.arange(t_bounds[0], t_bounds[1]), a)
result

# %%
spread = np.zeros((result.y.shape[0], 2, result.y.shape[1]))
spread[:, 0, :] = np.arange(x_bounds[0], x_bounds[1]).reshape(-1, 1)
spread[:, 1, :] = result.y
with open('spread.pkl', 'wb') as f:
    pickle.dump(spread, f)
spread

# %%
