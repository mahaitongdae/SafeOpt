from __future__ import print_function, division, absolute_import
import GPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import safeopt
# from safeopt.multi_agent import safeopt.MAGPRegression

mpl.rcParams['figure.figsize'] = (20.0, 10.0)
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.markersize'] = 20

# Measurement noise
noise_var = 0.05 ** 2
noise_var2 = 1e-5

# Bounds on the inputs variable

def given_safe_fun():
    bird = safeopt.benchmark_functions_2D.Bird()
    disk = safeopt.benchmark_functions_2D.Disk()
    disk.set_domain(bird)
    fun = lambda x: - bird.function(x)
    fun2 = lambda x: disk.function(x)
    arg_max = bird.arg_min
    bounds = disk.domain

    def combined_fun(x, noise=True):
        return np.hstack([fun(x), fun2(x)])
    return combined_fun, arg_max, bounds

# Define the objective function
# fun = sample_safe_fun()
fun, arg_max, bounds = given_safe_fun()

# Define Kernel
kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2., lengthscale=1.0, ARD=True)
kernel2 = GPy.kern.RBF(input_dim=len(bounds), variance=2., lengthscale=1.0, ARD=True)

# set of parameters
parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)

# Initial safe point
x0 = np.array([0., 0.])

# Communication network
N = np.eye(3)
N[0, 1] = N[1, 0] = N[1, 2] = N[2, 1] = 1
# N = np.ones([2,2])
n_workers = N.shape[0]

# The statistical model of our objective function and safety constraint
y0 = fun(x0)
models = [[safeopt.MAGPRegression(x0[np.newaxis, :], y0[0, np.newaxis, np.newaxis], kernel, noise_var=noise_var),
           safeopt.MAGPRegression(x0[np.newaxis, :], y0[1, np.newaxis, np.newaxis], kernel2, noise_var=noise_var2)] for i in range(n_workers)]

# The optimization routine
maopt = safeopt.MultiAgentSafeOptSwarm(N, fun, models, [-np.inf, 0.], bounds=bounds,
                                       threshold=0.2)

# maopt.optimize_with_different_tasks()
max_ys = []
for i in range(50):
    max_y = maopt.optimize()
    max_ys.append(max_y)
    print('iteration: {}, regret: {:.3e}'.format(i, maopt.fun(arg_max[0])[0] - max(max_ys)))
maopt.agents[0].plot(100, plot_3d=False)
plt.show()

