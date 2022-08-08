from __future__ import print_function, division, absolute_import
import GPy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import safeopt
import argparse
from safeopt.benchmark_functions_2D import *
# from safeopt.multi_agent import safeopt.MAGPRegression

mpl.rcParams['figure.figsize'] = (20.0, 10.0)
mpl.rcParams['font.size'] = 20
mpl.rcParams['lines.markersize'] = 20

# Measurement noise
noise_var = 0.05 ** 2
noise_var2 = 1e-5

# Bounds on the inputs variable

parser = argparse.ArgumentParser()
parser.add_argument('--objective', type=str, default='ackley')
parser.add_argument('--constraint', type=str, default='disk')
# parser.add_argument('--arg_max', type=np.ndarray, default=None)
parser.add_argument('--n_workers', type=int, default=3)
parser.add_argument('--kernel', type=str, default='RBF')
parser.add_argument('--acquisition_function', type=str, default='safeopt')
parser.add_argument('--policy', type=str, default='greedy')
parser.add_argument('--fantasies', type=int, default=0)
parser.add_argument('--regularization', type=str, default=None)
parser.add_argument('--regularization_strength', type=float, default=0.01)
parser.add_argument('--pending_regularization', type=str, default=None)
parser.add_argument('--pending_regularization_strength', type=float, default=0.01)
parser.add_argument('--grid_density', type=int, default=30)
parser.add_argument('--n_iters', type=int, default=50)
parser.add_argument('--n_runs', type=int, default=1)
args = parser.parse_args()

def given_safe_fun(objective):
    if objective == 'bird':
        obj = Bird()
    elif objective == 'ackley':
        obj = Ackley()
    elif objective == 'Rosenbrock':
        obj = Rosenbrock()
    elif objective == 'Bohachevsky':
        obj = Bohachevsky()
    elif objective == 'Rastrigin':
        obj = Rastrigin()
    elif objective == 'Himmelblau':
        obj = Himmelblau()
    elif objective == 'Eggholder':
        obj = Eggholder()
    elif objective == 'GoldsteinPrice':
        obj = GoldsteinPrice()
    elif objective == 'Branin':
        obj = Branin()
    else:
        raise NotImplementedError
    disk = Disk()
    disk.set_domain(obj)
    fun = lambda x: - obj.function(x)
    fun2 = lambda x: disk.function(x)
    arg_max = obj.arg_min
    bounds = disk.domain

    def combined_fun(x, noise=True):
        return np.hstack([fun(x), fun2(x)])
    return combined_fun, arg_max, bounds

# Define the objective function
# fun = sample_safe_fun()
fun, arg_max, bounds = given_safe_fun(args.objective)

# Define Kernel
kernel = GPy.kern.RBF(input_dim=len(bounds), variance=2., lengthscale=1.0, ARD=True)
kernel2 = GPy.kern.RBF(input_dim=len(bounds), variance=2., lengthscale=1.0, ARD=True)

# # set of parameters
# parameter_set = safeopt.linearly_spaced_combinations(bounds, 1000)

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
maopt = safeopt.MultiAgentSafeOptSwarm(N, fun, models, [-np.inf, 0.], arg_max, bounds=bounds,
                                       threshold=0.2,args=args)

# maopt.optimize_with_different_tasks()
maopt.optimize()
maopt.agents[0].plot(100, plot_3d=False)
plt.show()

