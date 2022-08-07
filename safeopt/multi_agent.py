from safeopt.gp_opt import SafeOptSwarm, SafeOpt
import GPy

class MultiAgentSafeOptSwarm(object):

    def __init__(self, network, fun, gp, fmin, bounds, threshold):
        self.network = network
        self.n_workers = network.shape[0]
        self.agents = []
        self.fun = fun
        self.ma_tasks = ['maximizers','expanders']
        assert len(self.ma_tasks) == self.n_workers
        for i in range(self.n_workers):
            self.agents.append(SafeOptSwarm(gp[i], fmin, bounds=bounds, threshold=threshold))

    def _broadcast(self,agent, x_next, y_meas):
        # broadcasting
        for neighbour_agent, neighbour in enumerate(self.network[agent]):
            if neighbour and neighbour_agent != agent:  #
                self.agents[neighbour_agent].add_new_data_point(x_next, y_meas, shared_data=True)


    def optimize(self):
        for agent_id, agent in enumerate(self.agents):
            x_next = agent.optimize()
            y_meas = self.fun(x_next)
            agent.add_new_data_point(x_next, y_meas)
            self._broadcast(agent_id, x_next, y_meas)

    def optimize_with_different_tasks(self):
        for agent_id, agent in enumerate(self.agents):
            x_next, _ = agent.get_new_query_point(self.ma_tasks[agent_id])
            y_meas = self.fun(x_next)
            agent.add_new_data_point(x_next, y_meas)
            self._broadcast(agent_id, x_next, y_meas)

class MAGPRegression(GPy.models.GPRegression):

    def __init__(self, X, Y, kernel=None, Y_metadata=None, normalizer=None, noise_var=1., mean_function=None):
        super(MAGPRegression, self).__init__(X, Y, kernel=kernel, Y_metadata=Y_metadata, normalizer=normalizer, noise_var=noise_var, mean_function=mean_function)
        self.data_sources=[True] # the first is added when initializing the GP

