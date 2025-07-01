import torch
from torch import nn, optim
import envs.custom_gyms
import gymnasium as gym
import math
import matplotlib.pyplot as plt
import copy
from lib.playground import play_episode, play_steps, ReplayBuffer


EPOCHS = 3000
ENV_SETTINGS = dict(id='custom/GridWorldGym', size=4, mode='random', max_moves=50, noise=0.01)
LEARN_RATE = .001
GAMMA = .9
EPS_GREEDY = .30
HIDDEN_SIZE = 128
REPLAY_SIZE = 1000 # steps
BATCH_SIZE = 256   # steps
DEVICE = 'cuda'

TARGET_NET_ENABLED = True
TARGET_NET_SYNC_STEPS = 50



class DQNAgent(nn.Module):
    def __init__(self, k_actions, n_states, hidden_size=HIDDEN_SIZE, eps=EPS_GREEDY):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, k_actions),
        )
        self.eps = eps
        self.k_actions = k_actions
        self.n_states = n_states

    def forward(self, state):
        state = state.view(state.shape[0], -1)      # B, *S -> B, S
        return self.net(state)                      # B, S  -> B, A

    @torch.no_grad()
    def policy(self, state, greedy=False):
        if not greedy and torch.rand(1) < self.eps:
            return torch.randint(self.k_actions, (1,)).item()

        state = torch.tensor(state, dtype=torch.float).view(1, -1).to(DEVICE)
        Q = self(state)
        return Q.argmax(dim=-1).item()


env = gym.make(**ENV_SETTINGS)
state_size = math.prod(env.observation_space.shape)
agent = DQNAgent(env.action_space.n, state_size).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(params=agent.parameters(), lr=LEARN_RATE)
replay = ReplayBuffer(capacity=REPLAY_SIZE)
if TARGET_NET_ENABLED:
    agent_target = copy.deepcopy(agent).requires_grad_(False)

# Initial replay buffer fill
print(f"Initial filling replay buffer:")
exp_iterator = play_steps(env, policy=agent.policy)
while len(replay) < BATCH_SIZE:
    obs, action, reward, obs_next, terminated, truncated = next(exp_iterator)
    replay.add(obs, action, reward, obs_next, terminated, truncated)
print(f"-> Replay buffer size: {len(replay)}/{replay.capacity}")


print(f"Training {EPOCHS} epochs. Target network: {TARGET_NET_ENABLED}")
mov_loss = 0
history = []
steps = 0
for epoch in range(1, EPOCHS+1):
    finished = False
    while not finished: # one epoch has one episode length steps
        # collect new experience
        obs, action, reward, obs_next, terminated, truncated = next(exp_iterator)
        replay.add(obs, action, reward, obs_next, terminated, truncated)
        finished = terminated or truncated

        # sample batched experiences from the replay buffer
        obs, actions, rewards, obs_next, done = replay.sample(batch_size=BATCH_SIZE, device=DEVICE)
        # done[rewards == -10] = 0  # for comparison (hacky)

        # compute future rewards
        with torch.no_grad(): # bootstrap
            if TARGET_NET_ENABLED:
                Q_next = agent_target(obs_next)
            else:
                Q_next = agent(obs_next)
        R = rewards + (1 - done) * GAMMA * Q_next.max(dim=-1, keepdim=True)[0]

        # update the model
        optimizer.zero_grad()
        Q_action = agent(obs).gather(dim=-1, index=actions)
        loss = loss_fn(Q_action, R.detach())
        loss.backward()
        optimizer.step()

        # collect some stats
        mov_loss = (.9 * mov_loss + .1 * loss.item()) if epoch > 1 else loss.item()
        history.append((loss.item(), mov_loss))
        steps += 1

        if TARGET_NET_ENABLED and steps % TARGET_NET_SYNC_STEPS == 0:
            agent_target.load_state_dict(agent.state_dict())


    if epoch % 100 == 0:
        print(f"#{epoch:>4} | {loss=:.6f}, {mov_loss=:.6f}, avg_rewards={rewards.mean():.2f} ")

        if epoch % 1000 == 0:
            n = 1000
            print(f"Testing over {n} games in random mode...")
            episodes = [play_episode(env, lambda s : agent.policy(s, greedy=True)) for _ in range(n)]
            env.reset() # because the same env is used for the replay buffer
            wins = sum([e.total_rewards > 0 for e in episodes])
            print(f"Win rate: {100 * wins / n:.1f}%")


# plot the loss
loss_, mov_loss_ = zip(*history)
plt.plot(loss_)
plt.plot(mov_loss_)
plt.xlabel("Steps"); plt.ylabel("Loss"); plt.tight_layout(); plt.show()


# Play one episode with visualization
print(f"Playing one episode with the trained agent")
env = gym.make(**ENV_SETTINGS, render_mode='human')
episode = play_episode(env, lambda s : agent.policy(s, greedy=True))
print(f"Episode finished with total reward {episode.total_rewards}")

