import gymnasium as gym
import numpy as np
import torch

from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from lib.playground import Episode, batched_episodes, play_episode
from lib.tracking import writer_add_params

ENV = 'CartPole-v1'
ENV_SOLVED_REWARD = 500
BATCH_SIZE = 16
PERCENTILE = 70
HIDDEN_SIZE = 128
LEARN_RATE = .01
LOG_STEP = 10


class Agent(nn.Module):

    def __init__(self, obs_size, n_actions):
        super(Agent, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, HIDDEN_SIZE),
            nn.ReLU(),
            nn.Linear(HIDDEN_SIZE, n_actions),
        )

    def forward(self, obs):
        return self.net(obs)

    def policy(self, state):
        z = self(torch.tensor(state, dtype=torch.float))
        p = torch.softmax(z, dim=-1)
        actions = torch.multinomial(p, num_samples=1).item()
        return actions


def filter_episodes(episodes, percentile=PERCENTILE):
    all_rewards = [ep.total_rewards for ep in episodes]
    avg_rewards = np.mean(all_rewards)
    threshold = np.percentile(all_rewards, percentile)
    elite_episodes = list(filter(lambda ep: ep.total_rewards >= threshold, episodes))
    return elite_episodes, avg_rewards, threshold



if __name__ == '__main__':
    env = gym.make(ENV, render_mode=None)
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(params=agent.net.parameters(), lr=LEARN_RATE)
    writer = SummaryWriter(comment=ENV)

    for i, batch in enumerate(batched_episodes(env, agent.policy, BATCH_SIZE)):
        elite, avg_rewards, reward_threshold = filter_episodes(batch)
        obs = torch.tensor(np.vstack([obs for episode in elite for obs in episode.observations]), dtype=torch.float)   # T, D
        actions = torch.tensor([actions for episode in elite for actions in episode.actions], dtype=torch.long)       # T

        optimizer.zero_grad()
        z = agent(obs)
        loss = loss_fn(z, actions)
        loss.backward()
        optimizer.step()

        if i % LOG_STEP == 0:
            print(f"#{i:>5}. loss={loss.item():.8f}, {avg_rewards=:.2f}, {reward_threshold=:.2f}, train_steps={len(actions)}")
            writer.add_scalar('t/Loss', loss, i)
            writer.add_scalar('t/Rewards', avg_rewards, i)
            writer.add_scalar('t/RewardsThreshold', reward_threshold, i)
            writer_add_params(writer, agent, i)

        if avg_rewards >= ENV_SOLVED_REWARD:
            print(f"Solved in {i} iterations!")
            env.close()
            break


    # Play one episode with visualization
    print(f"Playing one episode with the trained agent")
    env = gym.make(ENV, render_mode='human')
    episode = play_episode(env, agent.policy)
    print(f"Episode finished with reward {episode.total_rewards}")
    env.close()
