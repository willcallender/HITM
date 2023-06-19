import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import stats

filename = "spread.pkl"
conf = 0.99

x_bounds = np.array([100, 250])
t_bounds = np.array([30, 40])
# t_bounds = np.array([30, 31])

with open(filename, "rb") as f:
    spread = pickle.load(f)
spread = np.array(spread, float)
print(spread.shape)

y_bounds = np.array([0, spread.max()], float)

width = x_bounds[1] - x_bounds[0]
maxL = width // 3
params_time = np.zeros((t_bounds[1] - t_bounds[0], 2))
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
y = np.sqrt(deviations / num_samples)


x = np.log(np.arange(maxL) + 1)
y = np.log(np.sqrt(deviations / num_samples))

slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# 99% confidence interval
alpha = 1 - conf # significance level
z_critical = np.abs(stats.norm.ppf(alpha / 2))  # z-score for two-tailed test
# Calculate confidence intervals for a and b
a_interval = (intercept - z_critical * std_err, intercept + z_critical * std_err)
b_interval = (slope - z_critical * std_err, slope + z_critical * std_err)
print("a Confidence Interval:", a_interval)
print("b Confidence Interval:", b_interval)

plt.title("Power Law Fit")
plt.ylabel("$log(W(L))$")
plt.xlabel("$log(L)$")
plt.plot(x, y, "o")
plt.plot(x, intercept + slope * x)
plt.legend(["Data", "Fit $y = {:.3f}x + {:.3f}$".format(slope, intercept)])
plt.text(0.1, 0.1, f"a {conf*100:.0f}% CI: {a_interval}\nb {conf*100:.0f}% CI: {b_interval}", transform=plt.gca().transAxes)
