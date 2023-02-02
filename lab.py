# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import eye, kron
from scipy.sparse.linalg import spsolve
from methods import *

# %% [markdown]
# # 1.2 Exercise

# %% [markdown]
# Define ODE and solver params

# %%
lamb = 2
pi = np.pi
f = lambda x, y: (2 + 2 * pi**2)*np.sin(pi*x)*np.sin(pi*y)
Ns = 2**np.arange(2, 10) # Number of discretization points

# exact solution
u = lambda x,y: np.sin(pi*x)*np.sin(pi*y)

# %% [markdown]
# Run solver for every value of N

# %%
apxs = []
exs = []
grids = []
l2err = np.zeros(len(Ns))
for i, N in enumerate(Ns):
    approx, x, y = Screened_Poisson_2D(lamb, N, f)
    exact = u(x, y)
    apxs.append(approx)
    exs.append(exact)
    grids.append((x, y))
    l2err[i] = np.linalg.norm((exact - approx), 2) / N

# %% [markdown]
# Plot numerical convergence study

# %%
plt.loglog(Ns, l2err, label='Numerical convergence')
plt.loglog(Ns, Ns**(-2.0), label='Expected convergence')
plt.title('Relative error vs. number of discretization points')
plt.xlabel('Number of discretization points')
plt.ylabel('Relative error')
plt.legend()
plt.show()

# %% [markdown]
# Estimate numerical order of convergence

# %%
order = np.polyfit(np.log(Ns), np.log(l2err), 1)[0]
print(f'Order of convergence: {-order}')

# %%
# Show all approximations for debugging

# for approx, exact, grid in zip(apxs, exs, grids):
#     min = np.min((approx, exact))
#     max = np.max((approx, exact))
#     plt.subplot(1, 3, 1)
#     plt.imshow(approx, vmin=min, vmax=max)
#     plt.subplot(1, 3, 2)
#     plt.imshow(exact, vmin=min, vmax=max)
#     plt.subplot(1, 3, 3)
#     plt.imshow(approx - exact, vmin=min, vmax=max)
#     plt.colorbar()
#     plt.show()
#     # input()
#     x, y = grid
#     N = x.shape[0]
#     plt.plot(x[N//2, :], approx[N//2, :], label='approx')
#     plt.plot(x[N//2, :], exact[N//2, :], label='exact')
#     plt.legend()
#     plt.show()
#     # input()


