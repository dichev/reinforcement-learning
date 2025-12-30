import random
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import TransformObservation

from envs.wrappers import OneHotWrapper
from lib.grids import is_reachable

WORLDS = {
    'dyna_maze_9x6':"""
        . . . . . . . # G
        . . # . . . . # .
        S . # . . . . # .
        . . # . . . . . .
        . . . . . # . . .
        . . . . . . . . .
    """,
    'shortcut_maze_9x6_A': """
        . . . . . . . . G
        . . . . . . . . .
        . . . . . . . . .
        . # # # # # # # #
        . . . . . . . . .
        . . . S . . . . .
    """,
    'shortcut_maze_9x6_B': """
        . . . . . . . . G
        . . . . . . . . .
        . . . . . . . . .
        . # # # # # # # .
        . . . . . . . . .
        . . . S . . . . .
    """,
    'random_4x4':   dict(rows=4,  cols=4,  n_walls=1, n_lava=1),
    'random_10x10': dict(rows=10, cols=10, n_walls=5, n_lava=6),
}

FLOOR, WALL, LAVA, GOAL, AGENT, START = 0, 1, 2, 3, 4, 5
GRID_MAP = {
    FLOOR: '.',
    WALL:  '#',
    LAVA:  '~',
    GOAL:  'G',
    AGENT: 'A',
    START: 'S',
}
GRID_MAP_INV = {str_:int_ for int_, str_ in GRID_MAP.items()}
ACTION_MAP = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
ACTION_ARROWS_MAP = { 0: '↑', 1: '→', 2: '↓', 3: '←' }

class TinyGrid(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, template:str, max_steps=None, fully_observable=True, render_mode=None):
        super().__init__()
        self.template = template
        self.grid, self.start_pos = self.template_to_grid(template)
        self.rows, self.cols = self.grid.shape

        self.pos = self.start_pos
        self.steps = 0
        self.max_steps = max_steps if max_steps is not None else 4 * (self.rows * self.cols)
        self.fully_observable = fully_observable
        self.render_mode = render_mode

        self.action_space = gym.spaces.Discrete(4)
        if self.fully_observable:
            self.observation_space = gym.spaces.Box(low=0, high=max(GRID_MAP), shape=(self.rows, self.cols), dtype=np.uint8)
        else:
            self.observation_space = gym.spaces.Discrete(self.rows * self.cols)


    def get_observation(self):
        if self.fully_observable:
            ob = self.grid.copy()
            ob[self.pos] = AGENT
        else:
            i, j = self.pos
            ob = i * self.cols + j
        return ob

    def reset(self, options=None, **kwargs):
        if options and options.get('template'):
            self.template = options['template']

        self.grid, self.start_pos = self.template_to_grid(self.template)
        self.pos, self.steps = self.start_pos, 0
        if self.render_mode == 'human':
            self.render(f'Environment Reset: {options}')
        return self.get_observation(), {}

    def step(self, action):
        self.steps += 1

        row, col = self.pos
        match ACTION_MAP[action]:
            case 'up':    row -= 1
            case 'right': col += 1
            case 'down':  row += 1
            case 'left':  col -= 1

        if 0 <= row < self.rows and 0 <= col < self.cols:
            if self.grid[row, col] != WALL:
                self.pos = (row, col)

        cur = self.grid[self.pos].item()
        ob = self.get_observation()
        terminated = cur == GOAL or cur == LAVA
        truncated = self.steps >= self.max_steps
        if   cur == GOAL: reward =  10
        elif cur == LAVA: reward = -10
        else:             reward = -1

        if self.render_mode == 'human':
            self.render(f'Step {self.steps}: {reward=:.2f}')

        return ob, reward, terminated, truncated, {}

    def template_to_grid(self, template:str):
        if template not in WORLDS: raise ValueError(f"Unknown world template '{template}'. Available: {list(WORLDS.keys())}")
        text_grid = WORLDS[template]
        if isinstance(text_grid, dict):
            text_grid = self.generate_random_world(**WORLDS[template])
        grid, start_pos = self.text_to_grid(text_grid)
        return grid, start_pos


    @staticmethod
    def text_to_grid(grid_template):
        lines = [line.split() for line in grid_template.strip().split('\n')]
        grid = np.array([[GRID_MAP_INV[s] for s in row] for row in lines], dtype=np.uint8)
        start_pos = tuple(np.argwhere(grid == START)[0])
        grid[start_pos] = FLOOR  # clean up the start position
        return grid, start_pos

    @staticmethod
    def grid_to_text(grid:np.ndarray):
        return '\n'.join(' '.join(GRID_MAP[v] for v in row) for row in grid)

    def render(self, msg=''):
        view = self.grid.copy()
        view[self.pos] = AGENT
        print(f'\n{msg}')
        print(self.grid_to_text(view))

    def generate_random_world(self, rows, cols, n_walls, n_lava, max_tries=500):
        coords = [(r, c) for r in range(rows) for c in range(cols)]

        for _ in range(max_tries):
            grid = np.full((rows, cols), FLOOR, dtype=np.uint8)
            points = random.sample(coords, 2 + n_walls + n_lava)
            start_pos, goal_pos = points.pop(), points.pop()
            grid[start_pos] = START
            grid[goal_pos] = GOAL
            for _ in range(n_lava):  grid[points.pop()] = LAVA
            for _ in range(n_walls): grid[points.pop()] = WALL

            if is_reachable(grid, start_pos, goal_pos, (WALL, LAVA)):
                return self.grid_to_text(grid)

        raise RuntimeError(f"Could not generate solvable grid in {max_tries} tries.")



def make_TinyGrid(template, noise=0., max_steps=None, fully_observable=True, render_mode=None):
    env = TinyGrid(template, max_steps, fully_observable, render_mode)

    if fully_observable:
        # Split each cell type into a separate channel (one-hot)
        env = OneHotWrapper(env, channels=(WALL, LAVA, GOAL, AGENT))   # (cells, rows, cols)

        # Add noise
        if noise > 0:
            env = TransformObservation(env, lambda obs: ((1 - noise) * obs + noise * np.random.rand(*obs.shape)).astype(np.float32), env.observation_space)

    return env


if __name__ == '__main__':
    import envs.custom_gyms
    from lib.playground import play_episode

    # test TinyGrid (static)
    env1 = gym.make('custom/TinyGrid', template='dyna_maze_9x6', fully_observable=True, render_mode='human')
    ob1, _ = env1.reset()
    env1.step(env1.action_space.sample())
    episode1 = play_episode(env1, lambda s: env1.action_space.sample())

    # test TinyRandomGrid
    env2 = gym.make('custom/TinyGrid', template='random_4x4', noise=0.1, fully_observable=True, render_mode='human')
    env2.reset()
    env2.step(env2.action_space.sample())
    ob2, _ = env2.reset()
    env2.step(env2.action_space.sample())
    episode2 = play_episode(env2, lambda s: env2.action_space.sample())

    # test grid world change on reset:
    env3 = gym.make('custom/TinyGrid', template='shortcut_maze_9x6_A', fully_observable=True, render_mode='human')
    env3.reset(options={'template': 'shortcut_maze_9x6_A'})
    env3.reset(options={'template': 'shortcut_maze_9x6_B'})
