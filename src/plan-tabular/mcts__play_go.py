import multiprocessing as mp
import time
import numpy as np
import pygame
import random, math
from rollout__worker import Worker


GAME_BOARD_SIZE       = 5 # 9 # 19
ITERATIONS            = 100
ROLLOUTS              = 100
UCT_EXPLORATION_COEFF = 0.4 # math.sqrt(2)
NUM_WORKERS           = mp.cpu_count()




def make_env(render_mode=None):
    from pettingzoo.classic import go_v5
    return go_v5.env(board_size=GAME_BOARD_SIZE, komi=7.5, render_mode=render_mode)



def UCT(node, child, c=UCT_EXPLORATION_COEFF): # Upper Confidence bound applied to Trees
    win_ratio = (child.scores / child.visits + 1) / 2  # normalize as we collect scores as {lose: -1, win: 1}
    exploitation = win_ratio if node.is_player_turn else 1 - win_ratio # the opponent must choose the move that is WORST for the player
    exploration = c * math.sqrt(math.log(node.visits) / child.visits)
    return exploitation + exploration


class MCTSNode:
    def __init__(self, actions, is_player_turn, is_terminal):
        self.children = {}
        self.parent = None
        self.is_terminal = is_terminal
        self.is_player_turn = is_player_turn # otherwise is the opponent
        self.scores  = 0 # always from the player's perspective
        self.visits = 0
        self.untried = list(actions)
        random.shuffle(self.untried) # to ensure random selection in expand

    def sample(self):
        assert len(self.untried)
        return self.untried.pop()

    def __repr__(self):
        return f"MCTSNode(scores={self.scores}, visits={self.visits}, is_terminal={self.is_terminal}, is_player={self.is_player_turn})"


class MCTSAgent: # Monte Carlo Tree Search

    def __init__(self, id):
        self.id = id
        self.root = None
        self.cache = None # for subtree reuse
        self.model = EnvModel()

    def tree_policy(self, node):
        actions = node.children.keys()
        return max(actions, key=lambda action: UCT(node, node.children[action]))

    def best_action(self, actions, action_history, pool, num_simulations):
        print(f'Player ({self.id}) is thinking.. ({ITERATIONS} MCTS iterations x {num_simulations} rollouts) ', end='')

        if self.cache and (opponent_action := action_history[-1]) in self.cache.children:
            self.root = self.cache.children[opponent_action]  # reuse cache after the last move of the opponent
            self.root.parent = None
            self.cache = None
        else:
            self.root = MCTSNode(actions, is_player_turn=True, is_terminal=False)

        for i in range(ITERATIONS):
            trajectory = list(action_history.copy())
            leaf = self.select(trajectory)
            if not leaf.is_terminal and leaf.untried:
                leaf = self.expand(leaf, trajectory)

            scores, visits = self.simulate(leaf, pool, num_simulations, trajectory)
            self.backup(leaf, scores, visits)

        best_action, _ = max(self.root.children.items(), key=lambda item: item[1].visits)
        self.cache = self.root.children[best_action] # to reuse the subtree
        return best_action

    def select(self, trajectory):
        node = self.root
        while not node.is_terminal and not node.untried:
            action = self.tree_policy(node)
            node = node.children[action]
            trajectory.append(action)
        return node

    def expand(self, leaf, trajectory):
        action = leaf.sample() # random selection as the untried actions

        trajectory.append(action)
        is_terminal, next_turn, legal_actions = self.model.predict(trajectory)
        child = MCTSNode(legal_actions, is_player_turn=next_turn==self.id, is_terminal=is_terminal)
        child.parent = leaf
        leaf.children[action] = child

        return child

    def simulate(self, leaf, pool, num_simulations, trajectory):
        if leaf.is_terminal:
            score = self.model.last_score
            visits = num_simulations
            return score[self.id] * visits, visits

        # run all tasks in parallel
        tasks = [trajectory for _ in range(num_simulations)]
        results = pool.map(Worker.rollout, tasks)

        # collect the scores (of the player only)
        scores  = sum(score[self.id] for score in results)
        visits = num_simulations

        return scores, visits

    def backup(self, leaf, scores, visits):
        node = leaf
        while node is not None:
            node.visits += visits
            node.scores += scores # collecting only the player score
            node = node.parent



class EnvModel:

    def __init__(self):
        self.sim_env = make_env() # used as a sampling env model
        self.last_score = None

    def predict(self, trajectory):
        self.sim_env.reset()
        for a in trajectory:  # slow but we cannot deepcopy pettingzoo's envs
            self.sim_env.step(a)
        self.last_score = self.sim_env.rewards

        ob, reward, term, trunc, info = self.sim_env.last()
        next_turn = self.sim_env.agent_selection
        legal_actions = np.where(ob['action_mask'] == 1)[0]
        return term, next_turn, legal_actions




if __name__ == '__main__':
    mp.freeze_support()

    from rollout__play_go import RandomAgent, RolloutAgent
    from lib.plots import plot_monte_carlo_search_tree

    env = make_env(render_mode="human")
    env.reset()
    player, opponent = env.agents
    agents = {
        player: MCTSAgent(id=player),
        opponent: RolloutAgent(id=opponent),
        # opponent: MCTSAgent(id=opponent),
    }
    history = []
    print(f'Playing Go {GAME_BOARD_SIZE}x{GAME_BOARD_SIZE} board: Player {player} (MCTS) vs Opponent {opponent} (Rollout)')

    with mp.Pool(processes=NUM_WORKERS, initializer=Worker.init_worker, initargs=(make_env,)) as pool:
        for agent_id in env.agent_iter():
            ob, reward, termination, truncation, info = env.last()
            if termination or truncation: break
            legal_actions = np.where(ob['action_mask'] == 1)[0]
            assert len(legal_actions) > 0, 'No legal actions left! The game should have been terminated'

            start = time.time()
            action = agents[agent_id].best_action(legal_actions, history, pool, num_simulations=ROLLOUTS)
            print(f'\t-> Action #{action} ({time.time() - start:.2f}s)')
            history.append(action)
            env.step(action)
            pygame.event.pump() # prevent game UI freezing
            if agent_id == player:
                plot_monte_carlo_search_tree(agents[player].root, title='Monte Carlo Tree Search (player)')

        print(f"Game ended. Player ({player}) reward: {env.rewards[player]} | Opponent ({opponent}) reward: {env.rewards[opponent]}")
        # env.close()
