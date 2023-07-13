import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from numba import jit, njit

n = 1000
conf = 0.99
# H = np.random.uniform(-1, 1, n)
# H = np.random.normal(0, 1, n)
# H = np.random.exponential(1, n)
# H = np.random.binomial(1, 0.5, n)
# H = np.random.poisson(1, n)
# H = np.random.gamma(1, 1, n)
# H = np.random.beta(1, 1, n)
# H = np.random.chisquare(1, n)
# H = np.random.triangular(0, 1, 1, n)
# H = np.random.lognormal(0, 1, n)

# H = np.cumsum(H)

x_bounds = np.array([0, 10])
width = x_bounds[1] - x_bounds[0]
maxL = int(width/3)
# maxL = 3

def err(H):
    x, y, slope, intercept, r_value, p_value, std_err, a_interval, b_interval = fit(H)
    return -slope #np.abs(slope - 0.7)

# @njit
def fit(H):
    deviations = np.zeros(maxL)
    num_samples = np.zeros(maxL)
    for L in range(1, maxL + 1):
        for i in range(x_bounds[0], x_bounds[0] + width - L):
            h = H[i : i + L + 1]
            W = np.mean((h - np.mean(h)) ** 2)
            assert np.isnan(W) == False
            deviations[L - 1] += W
            num_samples[L - 1] += 1


    x = np.log(np.arange(maxL) + 1)
    y = np.log(np.sqrt(deviations / num_samples))

    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    # print(f"y = {slope}x + {intercept}")
    # print(f"r^2 = {r_value ** 2}")
    # print(f"p = {p_value}")
    # print(f"std_err = {std_err}")

    # confidence interval
    alpha = 1 - conf # significance level
    z_critical = np.abs(stats.norm.ppf(alpha / 2))  # z-score for two-tailed test
    # Calculate confidence intervals for a and b
    a_interval = (intercept - z_critical * std_err, intercept + z_critical * std_err)
    b_interval = (slope - z_critical * std_err, slope + z_critical * std_err)

    # print("a Confidence Interval:", a_interval)
    # print("b Confidence Interval:", b_interval)
    return x, y, slope, intercept, r_value, p_value, std_err, a_interval, b_interval

def plot(x, y, slope, intercept, a_interval, b_interval):
    # plot
    plt.clf()
    plt.subplot(211)
    plt.title(f"Power Law Fit")
    plt.ylabel("$log(W(L))$)")
    plt.xlabel("$log(L)$")
    plt.plot(x, y, "o")
    plt.plot(x, intercept + slope * x)
    plt.legend(["Data", "Fit $y = {:.3f}x + {:.3f}$".format(slope, intercept)], loc="upper left")
    plt.text(0.1, 0.1, f"b {conf*100:.0f}% CI: {b_interval}", transform=plt.gca().transAxes)

    # Plot the fireline
    plt.subplot(212)
    plt.title("Fireline")
    plt.ylabel("h")
    plt.xlabel("x")
    plt.plot(H)

    # Format and save
    plt.subplots_adjust(hspace=0.5)
    plt.show()

# H = np.sin(1 * np.pi * np.linspace(0, 10, n) / 10)
x = np.linspace(0, 10, n)
# H = 1 * x * (10 - x)
# H = 1/10 * (17 *L**6 - (101*L**5)/9 - (232*L**4)/9 + (1000 L^3)/9 + (10000 L^2)/3 - (20000 L)/3 + 10000/3) sqrt(x)
x, y, slope, intercept, r_value, p_value, std_err, a_interval, b_interval = fit(H)
plot(x, y, slope, intercept, a_interval, b_interval)

def genErr(x, rand):
    np.random.seed(0)
    H = rand(*x, n)
    # H = np.cumsum(H)
    # remove linear trend
    m, b = np.polyfit(np.arange(n), H, 1)
    H = H - (m * np.arange(n) + b)
    return err(H)

# rand = np.random.uniform
# x0 = np.array([-1, 1])

# rand = np.random.normal
# x0 = np.array([0, 1])

# rand = np.random.exponential
# x0 = np.array([1])

# ¯\_(ツ)_/¯
# rand = np.random.binomial
# x0 = np.array([0.5])

# rand = np.random.poisson
# x0 = np.array([1])

# ¯\_(ツ)_/¯
# rand = np.random.gamma
# x0 = np.array([1, 1])

# rand = np.random.beta
# x0 = np.array([1, 1])

# rand = np.random.chisquare
# x0 = np.array([1])

# ¯\_(ツ)_/¯
# rand = np.random.triangular
# x0 = np.array([0, 1, 1])

# rand = np.random.lognormal
# x0 = np.array([0, 1])

# minResult = minimize(genErr, x0, (rand))
# print(minResult)

# H = rand(*minResult.x, n)
# # H = np.cumsum(H)
# m, b = np.polyfit(np.arange(n), H, 1)
# H = H - (m * np.arange(n) + b)
# x, y, slope, intercept, r_value, p_value, std_err, a_interval, b_interval = fit(H)
# plot(x, y, slope, intercept, a_interval, b