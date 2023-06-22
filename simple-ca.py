import numpy as np
from matplotlib import pyplot as plt
from numba import jit, njit
import time
import pickle


n = 100
T = 100
ignition_prob = 0.5

@njit(fastmath=True)
def burn(fire, fuel, ignition_chance=ignition_prob):
    # each burning cell has a chance to ignite its neighbors
    # if they are not resistant to fire

    # find burning cells
    burning = np.argwhere(fire == 1)
    for cell in burning:
        # find neighbors
        x, y = cell
        neighbors = []
        if x > 0:
            neighbors.append((x - 1, y))
        if x < n - 1:
            neighbors.append((x + 1, y))
        if y > 0:
            neighbors.append((x, y - 1))
        if y < n - 1:
            neighbors.append((x, y + 1))
        
        # ignition chance 
        ignite = np.random.rand(len(neighbors))
        for i, neighbor in enumerate(neighbors):
            x, y = neighbor
            if fuel[x, y] != 1:
                fire[x, y] = (ignite[i] < ignition_chance) or (fire[x, y])
    return fire

start = time.time()

spread = np.zeros((T, 2, n))
fire = np.zeros((n, n))
fire[0, :] = 1
fuel = np.random.rand(n, n)
fuel = fuel < 0.1 # 10% chance of being resistant to fire

for t in range(T):
    fire = burn(fire, fuel)
    for x in range(n):
        for y in range(n):
            if fire[y, x] == 1:
                spread[t, 1, x] = y

end = time.time()
print(end - start)
print(spread[-1, 1, :])

# plt.imshow(fire, origin='lower')
# plt.show()
# plt.clf()

pickle.dump(spread, open("spread.pkl", "wb"))
