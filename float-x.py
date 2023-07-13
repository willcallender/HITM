import numpy as np
import matplotlib.pyplot as plt

from inverse import runExperiment

# Generate initial fireline
n = 100
spread = np.empty((2, 2, n))
spread[0, 1, :] = np.zeros(n)
spread[1, 1, :] = np.random.normal(1, 0.1, n)
spread[:, 0, :] = np.linspace(0, 1, n)

# plt.plot(spread[0, 0, :], spread[0, 1, :], label="Initial fireline")
# plt.plot(spread[1, 0, :], spread[1, 1, :], label="Final fireline")
# plt.show()

# runExperiment(spread, 0, 1, np.array([0, 1]), np.array([0, 1]))
h0, h1, verifyh, ha_result, y_bounds, finals, errs = runExperiment(spread, 0, 1, np.array([1]))

# Plot results
plt.plot(h0, 'b--')
plt.plot(h1, 'r.')
plt.plot(finals[0].y[:, -1], 'g-')
plt.legend(['first', 'last', 'approx'])
plt.show()
