import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm

filename = "spread.pkl"
with open(filename, "rb") as f:
    spread = pickle.load(f)
spread = np.array(spread, float)
print(spread.shape)

x_bounds = np.array([100, 250])
t_bounds = np.array([10, 80])
y_bounds = np.array([0, spread.max()], float)

width = x_bounds[1] - x_bounds[0]
maxL = width // 3
# sweep through length scales
deviations = np.zeros(maxL)
num_samples = np.zeros(maxL)
for t in range(t_bounds[0], t_bounds[1]):
    for L in range(1, maxL + 1):
        for i in range(x_bounds[0], x_bounds[0] + width - L):
            h = spread[t, 1, i : i + L + 1]
            W = np.mean((h - np.mean(h)) ** 2)
            deviations[L - 1] += W
            num_samples[L - 1] += 1

def power_law(x, a, b):
    return a * x ** b

x = np.arange(maxL)# + 1
y = deviations / num_samples

# print(deviations / num_samples)

# Fit the data to the power law function
params, cov = curve_fit(power_law, x, y)

# Extract the parameters and their uncertainties
a, b = params
sigma_a, sigma_b = np.sqrt(np.diag(cov))

# Calculate confidence intervals
conf = 0.95
alpha = 1 - conf # significance level
z_critical = np.abs(norm.ppf(alpha / 2))  # z-score for two-tailed test

# Calculate confidence intervals for a and b
a_interval = (a - z_critical * sigma_a, a + z_critical * sigma_a)
b_interval = (b - z_critical * sigma_b, b + z_critical * sigma_b)

print("a Confidence Interval:", a_interval)
print("b Confidence Interval:", b_interval)

# Plot the fitted function
plt.title("Power Law Fit")
plt.ylabel("W(L)")
plt.xlabel("L")
plt.loglog(y)
plt.loglog(power_law(x, *params))
plt.legend(["Data", "Fit $y = {:.3f}x^{{{:.3f}}}$".format(a, b)])
plt.text(0.1, 0.1, f"a {conf*100:.0f}% CI: {a_interval}\nb {conf*100:.0f}% CI: {b_interval}", transform=plt.gca().transAxes)

plt.show()
