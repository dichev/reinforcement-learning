import random
import gymnasium as gym
import numpy as np
import torch

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from lib.playground import batched_episodes, play_episode
from lib.tracking import writer_add_params
import envs.custom_gym

class DefaultConfig:
    PERCENTILE = 70
    HIDDEN_SIZE = 128
    LOG_STEP = 10

class FrozenLakeConfig(DefaultConfig):
    ENV = 'custom/FrozenLake-OneHot-DiscountedReward'
    ENV_SOLVED_REWARD = .95
    PRESERVE_ELITE = True
    BATCH_SIZE = 100
    LEARN_RATE = .001

class CartPoleConfig(DefaultConfig):
    ENV = 'CartPole-v1'
    ENV_SOLVED_REWARD = 500
    PRESERVE_ELITE = True
    BATCH_SIZE = 16
    LEARN_RATE = .01

cfg = CartPoleConfig
# cfg = FrozenLakeConfig


class Agent(nn.Module):

    def __init__(self, state_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, cfg.HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(cfg.HIDDEN_SIZE, n_actions),
        )

    def forward(self, state):
        return self.net(state)

    @torch.no_grad()
    def policy(self, state):
        z = self(torch.tensor(state, dtype=torch.float))
        p = torch.softmax(z, dim=-1)
        actions = torch.multinomial(p, num_samples=1).item()
        return actions


def filter_episodes(episodes, percentile=cfg.PERCENTILE):
    all_rewards = [ep.total_rewards for ep in episodes]
    threshold = np.percentile(all_rewards, percentile)
    elite_episodes = list(filter(lambda ep: ep.total_rewards >= threshold and ep.total_rewards > 0, episodes))
    if not elite_episodes: # always return at least one episode
        elite_episodes = [random.choice(episodes)]

    details = {
        "avg_rewards": np.mean(all_rewards),
        "max_rewards": np.max(all_rewards),
        "threshold": threshold,
    }
    return elite_episodes, details



if __name__ == '__main__':
    env = gym.make(cfg.ENV, render_mode=None)
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=agent.net.parameters(), lr=cfg.LEARN_RATE)
    writer = SummaryWriter(comment=cfg.ENV)

    elite_preserved = []
    for i, batch in enumerate(batched_episodes(env, agent.policy, cfg.BATCH_SIZE)):
        elite, details = filter_episodes(batch)
        if cfg.PRESERVE_ELITE:
            elite, _ = filter_episodes(elite + elite_preserved)
            elite_preserved = elite = elite[:cfg.BATCH_SIZE]

        obs = torch.cat([ep.observations for ep in elite])   # T, D
        actions = torch.cat([ep.actions for ep in elite])    # T

        optimizer.zero_grad()
        z = agent(obs)
        loss = loss_fn(z, actions)
        loss.backward()
        optimizer.step()

        if i % cfg.LOG_STEP == 0:
            print(f"#{i:>5}. loss={loss.item():.8f}, avg_rewards={details['avg_rewards']:.2f}, max_rewards={details['max_rewards']:.2f}, threshold={details['threshold']:.2f}, elite_episodes={len(elite)}/{len(batch)}, train_steps={len(actions)}")
            writer.add_scalar('t/Loss', loss, i)
            writer.add_scalar('t/Rewards', details['avg_rewards'], i)
            writer.add_scalar('t/RewardsThreshold', details['threshold'], i)
            writer_add_params(writer, agent, i)

        if details['avg_rewards'] >= cfg.ENV_SOLVED_REWARD:
            print(f"Solved in {i} iterations!")
            env.close()
            break


    # Play one episode with visualization
    print(f"Playing one episode with the trained agent")
    env = gym.make(cfg.ENV, render_mode='human')
    episode = play_episode(env, agent.policy)
    print(f"Episode finished with reward {episode.total_rewards}")
    env.close()
