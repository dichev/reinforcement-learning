import multiprocessing as mp
import time
import numpy as np
import pygame
from collections import defaultdict
from agents.planning.rollout__worker import Worker


GAME_BOARD_SIZE    = 9 # 19
ROLLOUTS           = 1000
NUM_WORKERS        = mp.cpu_count()


def make_env(render_mode=None):
    from pettingzoo.classic import go_v5
    return go_v5.env(board_size=GAME_BOARD_SIZE, render_mode=render_mode)


# def make_env(render_mode=None):
#     from pettingzoo.classic import chess_v6
#     return chess_v6.env(render_mode=render_mode)


class RolloutAgent:
    def __init__(self, id=None):
        self.id = id

    def best_action(self, actions, action_history, pool, num_simulations):
        num_actions = len(actions)
        print(f'Player ({self.id}) is thinking.. ({ROLLOUTS} rollouts over {num_actions} legal actions) ', end='')

        # select actions ensuring even coverage + uniform for the leftover
        selected_actions = np.concatenate([
            np.tile(actions, num_simulations // num_actions),
            np.random.choice(actions, num_simulations % num_actions, replace=False)
        ])
        np.random.shuffle(selected_actions)

        # run all tasks in parallel
        tasks = [action_history + [a] for a in selected_actions.tolist()]
        results = pool.map(Worker.rollout, tasks)

        # stores (only) the used actions
        action_values = defaultdict(float)
        visits = defaultdict(int)
        for action, scores in zip(selected_actions, results):
            action_values[action] += scores[self.id]
            visits[action] += 1

        # select the best action (no tie breaking as the order of action keys is random)
        best = max(action_values, key=lambda a: action_values[a]/visits[a])
        return best


class RandomAgent:
    def __init__(self, id=None):
        self.id = id

    def best_action(self, actions, *args, **kwargs):
        print(f'Opponent ({self.id}) selects a random action:', end='')
        action = np.random.choice(actions)
        return action



if __name__ == '__main__':
    mp.freeze_support()

    env = make_env(render_mode="human")
    env.reset()
    player, opponent = env.agents
    agents = {
        player: RolloutAgent(id=player),
        opponent: RandomAgent(id=opponent)
    }
    history = []
    print(f'Playing Go {GAME_BOARD_SIZE}x{GAME_BOARD_SIZE} board: Player {player} (rollout) vs Opponent {opponent} (uniform)')

    with mp.Pool(processes=NUM_WORKERS, initializer=Worker.init_worker, initargs=(make_env,)) as pool:
        for agent_id in env.agent_iter():
            ob, reward, termination, truncation, info = env.last()
            if termination or truncation: break
            legal_actions = np.where(ob['action_mask'] == 1)[0]
            assert len(legal_actions) > 0, 'No legal actions left! The game should have been terminated'

            start = time.time()
            action = agents[agent_id].best_action(legal_actions, history, pool, num_simulations=ROLLOUTS)
            print(f'-> Action #{action} ({time.time() - start:.2f}s)')
            history.append(action)
            env.step(action)
            pygame.event.pump() # prevent game UI freezing

    print(f"Game ended. Player ({player}) reward: {env.rewards[player]} | Opponent ({opponent}) reward: {env.rewards[opponent]}")
    # env.close()