import gymnasium as gym
import torch

from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from lib.playground import play_episode
from lib.tracking import writer_add_params
from lib.calc import discount_n_steps
from lib.utils import now

class cfg:
    EPOCHS = 3000
    LR_ACTOR  = .01
    LR_CRITIC = .001
    N_STEPS_BOOTSTRAP = 10 # bootstrap after N steps
    GAMMA = .99  # 1 is actually better for such episodic tasks
    HIDDEN_SIZE = 128
    ENV = 'CartPole-v1'
    ENV_GOAL = 480  # avg reward


class Actor(nn.Module):

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


class Critic(nn.Module):
    def __init__(self, n_states):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, cfg.HIDDEN_SIZE),
            nn.LeakyReLU(),
            nn.Linear(cfg.HIDDEN_SIZE, 1),
        )
    def forward(self, state):
        return self.net(state)


if __name__ == '__main__':
    env = gym.make(cfg.ENV, render_mode=None)
    actor = Actor(env.action_space.n, env.observation_space.shape[0])
    optimizer_actor = optim.Adam(params=actor.net.parameters(), lr=cfg.LR_ACTOR)
    critic = Critic(env.observation_space.shape[0])
    optimizer_critic = optim.Adam(params=critic.parameters(), lr=cfg.LR_CRITIC)
    writer = SummaryWriter(f'runs/A2C n_step={cfg.N_STEPS_BOOTSTRAP} {now()}', flush_secs=2)

    avg_reward = 0
    for epoch in range(1, cfg.EPOCHS+1):
        episode = play_episode(env, actor.policy)
        obs, actions, rewards, obs_next, done = episode.as_tensors()

        # Train the critic
        optimizer_critic.zero_grad()
        values = critic(obs)
        n, T = cfg.N_STEPS_BOOTSTRAP, episode.steps
        returns = discount_n_steps(rewards, cfg.GAMMA, n)                      # ignores rewards past n future steps
        if n <= T:                                                             # G_t = [R_{t+1} + γ R_{t+2} + ... + γ^{n-1} R_{t+n}] + γ^n V(s_{t+n})
            bootstrap = values[n:].detach()                                    # lookup 1 step ahead for each step
            returns[:T-n] +=  (cfg.GAMMA ** n) * bootstrap * (1 - done[:T-n])  # note: assumes that only the last step can be terminal: assert (done[:T-1]==0).all()
        loss_critic = F.mse_loss(values, returns)
        loss_critic.backward()
        optimizer_critic.step()


        # Train the actor (with N-step critic value)
        returns, baseline = returns.detach(), values.detach()   # the advantage must be treated as a constant, so detach them just in case (the actor's gradients must not flow to the critic)
        optimizer_actor.zero_grad()
        z = actor(obs)                                  # T, A
        logp = F.log_softmax(z, dim=-1)                 # T, A
        logp_a = logp.gather(-1, actions)               # T, 1
        loss_actor = -(logp_a * (returns - baseline)).mean()
        loss_actor.backward()
        optimizer_actor.step()


        avg_reward = (.9 * avg_reward + .1 * episode.total_rewards) if epoch > 1 else episode.total_rewards
        writer.add_scalar('t/Total reward (also episode length)', episode.total_rewards, epoch)
        writer.add_scalar('t/Return at step 0', returns[0], epoch)
        if epoch % 100 == 0:
            print(f"#{epoch:>5}. loss_actor={loss_actor.item():.4f}, loss_critic={loss_critic.item():.4f} last_reward={episode.total_rewards:.2f}, avg_reward={avg_reward:.2f}")
            writer.add_scalar('t/Loss Actor', loss_actor, epoch)
            writer.add_scalar('t/Loss Critic', loss_critic, epoch)
            writer.add_histogram('hist/Rewards', rewards, epoch)
            writer.add_histogram('hist/Observations', obs, epoch)
            writer.add_histogram('hist/Returns', returns, epoch)
            writer.add_histogram('hist/Baseline', baseline, epoch)
            writer.add_histogram('hist/Advantages', returns - baseline, epoch)
            writer_add_params(writer, actor, epoch)
            writer_add_params(writer, critic, epoch)

        if avg_reward >= cfg.ENV_GOAL:
            print(f"Solved in {epoch} epochs! Goal: {cfg.ENV_GOAL} Avg Reward = {avg_reward}")
            env.close()
            break


    # Play one episode with visualization
    print(f"Playing one episode with the trained agent")
    env = gym.make(cfg.ENV, render_mode='human')
    episode = play_episode(env, actor.policy)
    print(f"Episode finished with reward {episode.total_rewards}")
    env.close()

