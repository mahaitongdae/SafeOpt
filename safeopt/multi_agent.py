from safeopt.gp_opt import SafeOptSwarm, SafeOpt, linearly_spaced_combinations
import GPy
import os
import datetime
import __main__
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import csv
import json

class MultiAgentSafeOptSwarm(object):

    def __init__(self, network, fun, gp, fmin, arg_max, bounds, threshold, args, use_swarm=False):
        self.network = network
        self.n_workers = network.shape[0]
        self.agents = []
        self.fun = fun
        self.args = args
        self.arg_max = arg_max
        self.ma_tasks = ['maximizers','expanders']
        for i in range(self.n_workers):
            if use_swarm:
                self.agents.append(SafeOptSwarm(gp[i], fmin, bounds=bounds, threshold=threshold))
            else:
                parameter_set = linearly_spaced_combinations(bounds, 100)
                self.agents.append(SafeOpt(gp[i], parameter_set, 0., lipschitz=None, threshold=threshold))
        self._DT_ = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        self._ROOT_DIR_ = os.path.dirname(os.path.dirname(__main__.__file__))
        self._TEMP_DIR_ = os.path.join(os.path.join(self._ROOT_DIR_, "temp"), self.args.objective)
        self._ID_DIR_ = os.path.join(self._TEMP_DIR_, self._DT_+'MA-safeopt') if args.n_workers > 1 else os.path.join(self._TEMP_DIR_, self._DT_+'SA-safeopt')
        self._DATA_DIR_ = os.path.join(self._ID_DIR_, "data")
        self._FIG_DIR_ = os.path.join(self._ID_DIR_, "fig")
        self._PNG_DIR_ = os.path.join(self._FIG_DIR_, "png")
        self._PDF_DIR_ = os.path.join(self._FIG_DIR_, "pdf")
        self._GIF_DIR_ = os.path.join(self._FIG_DIR_, "gif")
        for path in [self._TEMP_DIR_, self._DATA_DIR_, self._FIG_DIR_, self._PNG_DIR_, self._PDF_DIR_, self._GIF_DIR_]:
            try:
                os.makedirs(path)
            except FileExistsError:
                pass

        # compute regret
        self.max_ys = []
        self.regret = []
        n_runs = self.args.n_runs
        n_iters = self.args.n_iters
        self._simple_regret = np.zeros((n_runs, n_iters + 1))
        self._distance_traveled = np.zeros((n_runs, n_iters + 1))


    def _broadcast(self,agent, x_next, y_meas):
        # broadcasting
        for neighbour_agent, neighbour in enumerate(self.network[agent]):
            if neighbour and neighbour_agent != agent:  #
                self.agents[neighbour_agent].add_new_data_point(x_next, y_meas, shared_data=True)


    def optimize(self):
        for run in range(self.args.n_runs):
            self.old_x = [None for i in range(len(self.agents))]
            for i in range(self.args.n_iters + 1):
                y_meass = []
                dist_traveled = 0.
                for agent_id, agent in enumerate(self.agents):
                    x_next = agent.optimize(ucb=self.args.ucb)
                    y_meas = self.fun(x_next)
                    agent.add_new_data_point(x_next, y_meas)
                    self._broadcast(agent_id, x_next, y_meas)
                    y_meass.append(y_meas[0])
                    if not i:
                        self._distance_traveled[run, i] = 0
                        self.old_x[agent_id] = x_next
                    else:
                        # print(self.old_x)
                        self._distance_traveled[run, i] = self._distance_traveled[run, i - 1] + sum(
                            [np.linalg.norm(x_next - self.old_x[agent_id]) for a in range(self.n_workers)])

                self.max_ys.append(max(y_meass)) # max y from each agent
                self._simple_regret[run, i] = self.fun(self.arg_max[0])[0] - max(self.max_ys)

                print('step: {}, current_max: {:.3f}, arg max: {:.3f}, regret: {:.3e}'.format(i, max(self.max_ys), self.fun(self.arg_max[0])[0], self._simple_regret[run, i]))

        iter, r_mean, r_conf95 = self._mean_regret()
        self._plot_regret(iter, r_mean, r_conf95)

        iter, d_mean, d_conf95 = self._mean_distance_traveled()

        self._save_data(data=[iter, r_mean, r_conf95, d_mean, d_conf95], name='data')

    def optimize_with_different_tasks(self):
        assert len(self.ma_tasks) == self.n_workers
        for agent_id, agent in enumerate(self.agents):
            x_next, _ = agent.get_new_query_point(self.ma_tasks[agent_id])
            y_meas = self.fun(x_next)
            agent.add_new_data_point(x_next, y_meas)
            self._broadcast(agent_id, x_next, y_meas)

    def _mean_regret(self):
        r_mean = [np.mean(self._simple_regret[:,iter]) for iter in range(self._simple_regret.shape[1])]
        r_std = [np.std(self._simple_regret[:,iter]) for iter in range(self._simple_regret.shape[1])]
        # 95% confidence interval
        conf95 = [1.96*rst/self._simple_regret.shape[0] for rst in r_std]
        return range(self._simple_regret.shape[1]), r_mean, conf95

    def _mean_distance_traveled(self):
        d_mean = [np.mean(self._distance_traveled[:,iter]) for iter in range(self._distance_traveled.shape[1])]
        d_std = [np.std(self._distance_traveled[:,iter]) for iter in range(self._distance_traveled.shape[1])]
        # 95% confidence interval
        conf95 = [1.96*dst/self._distance_traveled.shape[0] for dst in d_std]
        return range(self._distance_traveled.shape[1]), d_mean, conf95

    def _plot_regret(self, iter, r_mean, conf95, log = False):

        use_log_scale = max(r_mean)/min(r_mean) > 10

        if not use_log_scale:
            # absolut error for linear scale
            lower = [r + err for r, err in zip(r_mean, conf95)]
            upper = [r - err for r, err in zip(r_mean, conf95)]
        else:
            # relative error for log scale
            lower = [10**(np.log10(r) + (0.434*err/r)) for r, err in zip(r_mean, conf95)]
            upper = [10**(np.log10(r) - (0.434*err/r)) for r, err in zip(r_mean, conf95)]

        fig = plt.figure()

        if use_log_scale:
            plt.yscale('log')

        plt.plot(iter, r_mean, '-', linewidth=1)
        plt.fill_between(iter, upper, lower, alpha=0.3)
        plt.xlabel('iterations')
        plt.ylabel('immediate regret')
        plt.grid(b=True, which='major', color='grey', linestyle='-', alpha=0.6)
        plt.grid(b=True, which='minor', color='grey', linestyle='--', alpha=0.3)
        plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
        plt.tight_layout()
        if use_log_scale:
            plt.savefig(self._PDF_DIR_ + '/regret_log.pdf', bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/regret_log.png', bbox_inches='tight')
        else:
            plt.savefig(self._PDF_DIR_ + '/regret.pdf', bbox_inches='tight')
            plt.savefig(self._PNG_DIR_ + '/regret.png', bbox_inches='tight')

    def _save_data(self, data, name):
        with open(self._DATA_DIR_ + '/config.json', 'w', encoding='utf-8') as file:
            json.dump(vars(self.args), file, ensure_ascii=False, indent=4)

        with open(self._DATA_DIR_ + '/' + name + '.csv', 'w', newline='') as file:
            writer = csv.writer(file, delimiter=',')
            for i in zip(*data):
                writer.writerow(i)

        return


class MAGPRegression(GPy.models.GPRegression):

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):
        super(MAGPRegression, self).__init__(X, Y, kernel=kernel, Y_metadata=Y_metadata, normalizer=normalizer, noise_var=noise_var, mean_function=mean_function)
        self.data_sources=[True] # the first is added when initializing the GP

