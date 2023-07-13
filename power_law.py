import numpy as np
import numpy.typing as npt
import pickle
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import norm
from scipy import stats

def PLfit(h, x_bounds: npt.NDArray[np.floating] | None = None, y_bounds: npt.NDArray[np.floating] | None = None, conf=0.99, plot=False):
    # auto determine bounds if not given
    if x_bounds is None:
        x_bounds = np.array([0, len(h)])
    if y_bounds is None:
        y_bounds = np.array([h.min(), h.max()])
    
    # sweep through length scales
    width = x_bounds[1] - x_bounds[0]
    maxL = width // 3
    deviations = np.zeros(maxL)
    num_samples = np.zeros(maxL)
    for L in range(1, maxL + 1):
        for i in range(x_bounds[0], x_bounds[0] + width - L):
            h_L = h[i : i + L + 1]
            W = np.mean((h_L - np.mean(h_L)) ** 2)
            deviations[L - 1] += W
            num_samples[L - 1] += 1

    # perform linear regression on log-log plot
    x = np.log(np.arange(maxL) + 1)
    y = np.log(np.sqrt(deviations / num_samples))

    # Fit the data to the power law function
    slope, intercept, r_value, p_value, stderr_slope = stats.linregress(x,y)

    # Extract the parameters and their uncertainties
    df = len(x) - 2
    alpha = 1 - conf
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    # Compute the confidence intervals for the slope
    slope_interval = (slope - t_crit * stderr_slope, slope + t_crit * stderr_slope)

    if plot:
        plt.title("Power Law Fit")
        plt.ylabel("$log(W(L))$")
        plt.xlabel("$log(L)$")
        plt.plot(x, y, "o")
        plt.plot(x, intercept + slope * x)
        plt.legend(["Data", "Fit $y = {:.3f}x + {:.3f}$".format(slope, intercept)])
        plt.text(0.1, 0.1, f"{conf*100:.0f}% CI: {slope_interval}", transform=plt.gca().transAxes)
        plt.show()


if __name__ == "__main__":    
    # filename = "spread.pkl"
    # conf = 0.99

    # x_bounds = np.array([100, 250])
    # t = 32

    # with open(filename, "rb") as f:
    #     spread = pickle.load(f)
    # spread = np.array(spread, float)
    # # print(spread.shape)

    # PLfit(spread[t, 1, :], x_bounds, conf=conf, plot=True)

    x = np.linspace(0, 1)
    h = x*(1-x)
    PLfit(h, plot=True)
