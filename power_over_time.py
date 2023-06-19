import numpy as np
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import stats


filename = "spread.pkl"
conf = 0.99
# x_bounds = np.array([100, 250])
# t_bounds = np.array([10, 50])
x_bounds = np.array([0, 100])
t_bounds = np.array([0, 100])

def power_law(x, a, b):
    return a * x ** b

with open(filename, "rb") as f:
    spread = pickle.load(f)
spread = np.array(spread, float)
print(spread.shape)

y_bounds = np.array([0, spread.max()], float)

width = x_bounds[1] - x_bounds[0]
maxL = width // 3
b_time = np.zeros(t_bounds[1] - t_bounds[0])
CI = np.zeros((t_bounds[1] - t_bounds[0], 2))
# sweep through length scales
for t in range(t_bounds[0], t_bounds[1]):
    deviations = np.zeros(maxL)
    num_samples = np.zeros(maxL)
    for L in range(1, maxL + 1):
        for i in range(x_bounds[0], x_bounds[0] + width - L):
            h = spread[t, 1, i : i + L + 1]
            W = np.mean((h - np.mean(h)) ** 2)
            deviations[L - 1] += W
            num_samples[L - 1] += 1


    x = np.log(np.arange(maxL) + 1)
    y = np.log(np.sqrt(deviations / num_samples))

    # Fit the data to the power law function
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    # print(t, params)

    # Extract the parameters and their uncertainties

    # 99% confidence interval
    alpha = 1 - conf # significance level
    z_critical = np.abs(stats.norm.ppf(alpha / 2))  # z-score for two-tailed test
    # Calculate confidence intervals for a and b
    a_interval = (intercept - z_critical * std_err, intercept + z_critical * std_err)
    b_interval = (slope - z_critical * std_err, slope + z_critical * std_err)

    b_time[t - t_bounds[0]] = slope
    CI[t - t_bounds[0]] = b_interval
    # print("a Confidence Interval:", a_interval)
    # print("b Confidence Interval:", b_interval)

    # Plot the fitted function
    plt.clf()
    plt.subplot(211)
    plt.title(f"Power Law Fit t={t}")
    plt.ylabel("$log(W(L))$)")
    plt.xlabel("$log(L)$")
    plt.plot(x, y, "o")
    plt.plot(x, intercept + slope * x)
    plt.legend(["Data", "Fit $y = {:.3f}x + {:.3f}$".format(slope, intercept)], loc="upper left")
    plt.text(0.1, 0.1, f"b {conf*100:.0f}% CI: {b_interval}", transform=plt.gca().transAxes)

    # Plot the current and original fireline
    plt.subplot(212)
    plt.title("Fireline")
    plt.ylabel("h")
    plt.xlabel("x")
    plt.plot(spread[t, 1]) # current
    plt.plot(spread[t_bounds[0], 1], "k--") # original
    plt.ylim(y_bounds)
    plt.xlim(x_bounds)

    # Format and save
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f"frames/{t-t_bounds[0]:03d}.png")
    # plt.show()

# # plot b vs time
# plt.clf()
# plt.plot(np.arange(t_bounds[0], t_bounds[1]), b_time)
# plt.fill_between(np.arange(t_bounds[0], t_bounds[1]), CI[:, 1], CI[:, 0], alpha=0.4)
# plt.title("Power Law Fit vs Time")
# plt.xlabel("Time")
# plt.ylabel("Slope ($b$)")
# plt.legend(["Fit"])
# plt.show()

from os import system
system("ffmpeg -y -framerate 1 -i frames/%03d.png -c:v libx264 -r 5 -pix_fmt yuv420p out.mp4")
