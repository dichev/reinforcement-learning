import numpy as np


class Worker:
    _instance = None

    def __init__(self, env):
        self.sim_env = env

    def _sync_env(self, action_history):
        self.sim_env.reset()
        for a in action_history: # slow but we cannot deepcopy pettingzoo's envs
            self.sim_env.step(a)

    def _rollout(self):  # play a full episode starting from the current state
        while self.sim_env.agent_iter():
            ob, reward, term, trunc, info = self.sim_env.last()
            if term or trunc:
                break
            legal_actions = np.where(ob['action_mask'] == 1)[0]
            action = np.random.choice(legal_actions)  # using random rollout policy
            self.sim_env.step(action)

        return self.sim_env.rewards


    @classmethod
    def init_worker(cls, make_env):
        env = make_env()
        cls._instance = cls(env)

    @classmethod
    def rollout(cls, action_history):
        self = cls._instance
        self._sync_env(action_history)
        scores = self._rollout()
        return scores
