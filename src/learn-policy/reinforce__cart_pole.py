import gymnasium as gym
import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.playground import play_episode
from lib.tracking import writer_add_params
from lib.calc import discount_returns
from lib.utils import now

class cfg:
    EPOCHS = 3000
    LEARN_RATE = .001
    GAMMA = .99  # 1 is actually better for such episodic tasks
    HIDDEN_SIZE = 128
    ENV = 'CartPole-v1'
    ENV_GOAL = 480  # avg reward


class PolicyAgent(nn.Module):

    def __init__(self, k_actions, n_states):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, cfg.HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(cfg.HIDDEN_SIZE, k_actions),
        )

    def forward(self, state):
        return self.net(state)

    @torch.no_grad()
    def policy(self, state):
        state = torch.tensor(state, dtype=torch.float)
        z = self(state)
        p = torch.softmax(z, dim=-1)
        action = torch.multinomial(p, num_samples=1).item()
        return action



if __name__ == '__main__':
    env = gym.make(cfg.ENV, render_mode=None)
    agent = PolicyAgent(env.action_space.n, env.observation_space.shape[0])
    optimizer = optim.Adam(params=agent.net.parameters(), lr=cfg.LEARN_RATE)
    writer = SummaryWriter(f'runs/REINFORCE baseline=mean {now()}', flush_secs=2)

    avg_reward = 0
    for epoch in range(1, cfg.EPOCHS+1):
        episode = play_episode(env, agent.policy)
        obs, actions, rewards, obs_next, done = episode.as_tensors()
        returns = discount_returns(rewards.view(-1, 1), cfg.GAMMA)
        baseline = returns.mean(dim=0)

        optimizer.zero_grad()
        z = agent(obs)                                  # T, A
        logp = F.log_softmax(z, dim=-1)                 # T, A
        logp_a = logp.gather(-1, actions.view(-1, 1))   # T, 1
        loss = -(logp_a * (returns - baseline)).mean()
        loss.backward()
        optimizer.step()

        avg_reward = (.9 * avg_reward + .1 * episode.total_rewards) if epoch > 1 else episode.total_rewards

        writer.add_scalar('t/Total reward (also episode length)', episode.total_rewards, epoch)
        writer.add_scalar('t/Return at step 0', returns[0], epoch)
        if epoch % 100 == 0:
            print(f"#{epoch:>5}. loss={loss.item():.4f}, last_reward={episode.total_rewards:.2f}, avg_reward={avg_reward:.2f}")
            writer.add_scalar('t/Loss', loss, epoch)
            writer.add_histogram('hist/Rewards', rewards, epoch)
            writer.add_histogram('hist/Observations', obs, epoch)
            writer.add_histogram('hist/Returns', returns, epoch)
            writer.add_histogram('hist/Baseline', baseline, epoch)
            writer.add_histogram('hist/Advantages', returns - baseline, epoch)
            writer_add_params(writer, agent, epoch)

        if avg_reward >= cfg.ENV_GOAL:
            print(f"Solved in {epoch} epochs! Goal: {cfg.ENV_GOAL} Avg Reward = {avg_reward}")
            env.close()
            break


    # Play one episode with visualization
    print(f"Playing one episode with the trained agent")
    env = gym.make(cfg.ENV, render_mode='human')
    episode = play_episode(env, agent.policy)
    print(f"Episode finished with reward {episode.total_rewards}")
    env.close()

