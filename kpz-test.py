from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
import pickle
from os import listdir, remove

# numerical solver for the Kardar-Parisi-Zhang equation
def kpz(h, t, c, dx, v, lamb, y=1):
    Gamma = np.empty_like(h)
    # second-order central
    Gamma[1:-1] = h[2:] + h[:-2] - 2*h[1:-1]
    # second order forward
    Gamma[0] = h[2] + h[0] - 2*h[1]
    # second order backward
    Gamma[-1] = Gamma[-1] + Gamma[-3] - 2*h[-2]
    
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

    return c + 1/(dx**2) * (v*Gamma + Psi*lamb*0.5)

# define constants, rounded from sec 5.2 in Campos 2013
c = 0.2
v = 0.5
lamb = 0.7
D = 5

# define init conds
timestamps = np.linspace(0, 10)
x = np.linspace(0, 1)
y0 = -(np.linspace(-1, 1)**2) + 1 # creates inverted parabola
# Y = np.zeros((*Y_0.shape, timestamps.size))
# plt.clf()
# plt.plot(Y_0) # show init cond

# function callable by scipy
def f(t, y):
    return kpz(y, t, c, 1, v, lamb)

# solve over time and plot
result = solve_ivp(f, [0, 1], y0, vectorized=True)
with open('result.dat', 'wb') as f:
    pickle.dump(result, f)

# delete all files from frames folder
for fname in listdir('frames'):
    remove(f'frames/{fname}')

for i in range(result.y.shape[1]):
    plt.clf()
    plt.ylim(0, 1.5)
    plt.plot(result.y[:,i])
    plt.savefig(f'./frames/{i:03d}.png')
plt.clf()
plt.plot(result.y)
plt.show()