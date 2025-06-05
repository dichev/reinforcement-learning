import gymnasium as gym
import numpy as np
import torch
import math
from torch import nn, optim
from torch.nn.utils import parameters_to_vector
from torch.utils.tensorboard import SummaryWriter


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


class Episode:
    def __init__(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.total_rewards = 0
        self.steps = 0

    def step(self, obs, action, reward):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.total_rewards += reward
        self.steps += 1

    def __repr__(self):
        return f'Episode(steps={self.steps}, total_rewards={self.total_rewards})'


@torch.no_grad()
def generate_episodes(env, policy, num_episodes):
    while True:
        batch = [play_episode(env, policy) for _ in range(num_episodes)]
        yield batch

def play_episode(env, policy):
    episode = Episode()
    obs, _ = env.reset()
    while True:
        action = policy(obs)
        obs_next, reward, terminated, truncated, _ = env.step(action)
        episode.step(obs, action, reward)
        obs = obs_next
        if terminated or truncated:
            return episode


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
    writer = SummaryWriter(comment="-cartpole")

    for i, batch in enumerate(generate_episodes(env, agent.policy, BATCH_SIZE)):
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
            writer.add_scalar('a/Gradients Norm',  parameters_to_vector(p.grad.norm() for p in agent.parameters()).norm(), i)
            writer.add_scalar('a/Weights Norm', parameters_to_vector(p.norm() for name, p in agent.named_parameters()).norm(), i)
            for name, param in agent.named_parameters():
                if 'bias' not in name:
                    writer.add_histogram('params/' + name.replace('.', '/'), param, i)
                    writer.add_histogram('grad/' + name.replace('.', '/'), param.grad, i)  # note this is a sample from the last mini-batch

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
