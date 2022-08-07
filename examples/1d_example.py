# %%

from __future__ import print_function, division, absolute_import

import GPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import safeopt

mpl.rcParams['figure.figsize'] = (20.0, 10.0)
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.markersize'] = 20

# %%

# Measurement noise
noise_var = 0.05 ** 2

# Bounds on the inputs variable
bounds = [(-10., 10.)]
parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)

# Define Kernel
kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2., lengthscale=1.0, ARD=True)

# Initial safe point
x0 = np.zeros((1, len(bounds)))


# Generate function with safe initial point at x=0
def sample_safe_fun():
    while True:
        fun = safeopt.sample_gp_function(kernel, bounds, noise_var, 100)
        if fun(0, noise=False) > 0.5:
            break
    return fun


# %% md

## Interactive run of the algorithm

# %%

# Define the objective function
fun = sample_safe_fun()

# The statistical model of our objective function
gp = GPy.models.GPRegression(x0, fun(x0), kernel, noise_var=noise_var)

# The optimization routine
# opt = safeopt.SafeOptSwarm(gp, 0., bounds=bounds, threshold=0.2)
opt = safeopt.SafeOpt(gp, parameter_set, 0., lipschitz=None, threshold=0.2)


# %%

def plot_gp():
    # Plot the GP
    opt.plot(1000)
    # Plot the true function
    plt.plot(parameter_set, fun(parameter_set, noise=False), color='C2', alpha=0.3)


# plot_gp()

# %%

# Obtain next query point
x_next = opt.optimize()
# Get a measurement from the real system
y_meas = fun(x_next)
# Add this to the GP model
opt.add_new_data_point(x_next, y_meas)

plot_gp()

# %%


# %%


# %%


# %%


# %%


