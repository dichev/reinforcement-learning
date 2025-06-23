import gymnasium as gym
import numpy as np
from gymnasium.envs.registration import register
from envs.external.GridWorld import GridWorld

class GridWorldGym(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, size=4, mode='static', max_moves=np.inf, noise=0, render_mode=None):
        self.size = size
        self.mode = mode
        self.render_mode = render_mode
        self.max_moves = max_moves
        self.noise = noise
        self.action_space = gym.spaces.Discrete(4)
        self.action_map = {0:'u', 1:'d', 2:'l', 3:'r'}
        self.observation_space = gym.spaces.Box(0.-self.noise, 1.+self.noise, (4, size, size), np.float64)

        self.game = GridWorld(size=size, mode=mode)
        self.steps = 0
        self.finished = False

    def step(self, action):
        if self.finished:
            raise Exception('Game already finished')

        action_str = self.action_map[action]
        self.game.makeMove(action_str)
        self.steps += 1
        obs_next = self.game.board.render_np() + self.gen_noise()
        reward = self.game.reward()
        terminated = reward != -1  # note the original game doesn't terminate on pit: reward > 0
        truncated = self.steps >= self.max_moves
        info = {}
        self.finished = terminated or truncated
        if self.render_mode == 'human': self.display(f'{reward=}')
        return obs_next, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.game = GridWorld(self.size, self.mode)
        self.steps = 0
        self.finished = False

        obs_next = self.game.board.render_np() + self.gen_noise()
        info = {}
        if self.render_mode == 'human': self.display('RESET')
        return obs_next, info

    def gen_noise(self):
        return np.random.rand(self.size, self.size) * self.noise if self.noise > 0 else 0


    def display(self, msg=''):
        print(f'\nStep {self.steps}: {msg}')
        print(self.game.display())


env_id = 'custom/GridWorldGym'
if env_id not in gym.envs.registry:
    register(id=env_id, entry_point=GridWorldGym)


if __name__ == '__main__':
    from lib.playground import play_episode
    # env = GridWorldGym(size=4, mode='random', noise=0.01, render_mode='human')
    env = gym.make('custom/GridWorldGym', size=4, mode='random', noise=0.01, render_mode='human')
    episode = play_episode(env, lambda s: np.random.randint(4))
