import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy.abc import x
from scipy import stats

def W(h, L):
    W = sp.sqrt(sp.integrate((h - (sp.integrate(h, (x, 0, L)) / L))**2, (x, 0, L)) / L)
    return W

eqns = [x, x**2]
for h in eqns:
    L = np.linspace(1, 10)
    W_vals = np.array([float(W(h, l)) for l in L])
    print(stats.linregress(np.log(L), np.log(W_vals)))
    plt.loglog(L, W_vals)
plt.legend(["$x$", "$x^2$"])
# plt.show()
plt.clf()

from power_law import PLfit
x = np.linspace(0, 10)
h = x
PLfit(h, plot=True)