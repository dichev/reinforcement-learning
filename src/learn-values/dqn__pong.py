import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
import envs.custom_gyms
import gymnasium as gym
import copy
import time
from lib.playground import play_episode, play_steps, ReplayBuffer, evaluate_policy_agent
from lib.tracking import writer_add_params
from lib.utils import now


ENV_SETTINGS = dict(id='custom/Pong')
ENV_GOAL = 19
LEARN_RATE = .0001
GAMMA = .99
EPS = dict(
    INITIAL = 1.0,
    FINAL = 0.05,
    DURATION = 150_000,
)
REPLAY_SIZE = 10_000
REPLAY_SIZE_START = 10_000
BATCH_SIZE = 32   # steps
LOG_STEP = 100
DEVICE = 'cuda'
TARGET_NET_ENABLED = True
TARGET_NET_SYNC_STEPS = 1_000
EVAL_NUM_EPISODES = 10



class DQNAgent(nn.Module):
    def __init__(self, k_actions, n_frames, eps=EPS['INITIAL'], use_target_net=TARGET_NET_ENABLED):
        super().__init__()
        self.net = nn.Sequential(                                                         # in:  n, 84, 84
            nn.Conv2d(in_channels=n_frames, out_channels=32, kernel_size=8, stride=4),    # ->  32, 20, 20
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),          # ->  64,  9,  9
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),          # ->  64,  7,  7
            nn.ReLU(),
            nn.Flatten(),                                                                 # -> 3136 (flatten)
            nn.Linear(64 * 7 * 7, 512),                                       # -> 512
            nn.ReLU(),
            nn.Linear(512, k_actions)                                          # -> k_actions
        )
        if use_target_net:
            self.target_net = copy.deepcopy(self.net)
            self.target_net.requires_grad_(False)

        self.k_actions = k_actions
        self.frames = n_frames
        self.register_buffer('eps', torch.tensor(eps))

    def forward(self, state):
        B, C, H, W = state.shape
        return self.net(state)  # B, A

    def sync_target_net(self):
        self.target_net.load_state_dict(self.net.state_dict())

    @torch.no_grad()
    def policy(self, state, greedy=False):
        if not greedy and torch.rand(1).item() < self.eps:
            return torch.randint(self.k_actions, (1,)).item()

        C, H, W = state.shape
        state = torch.tensor(state, dtype=torch.float).view(1, C, H, W).to(DEVICE)
        Q = self(state)
        return Q.argmax(dim=-1).item()


env = gym.make(**ENV_SETTINGS)
agent = DQNAgent(env.action_space.n, env.observation_space.shape[0]).to(DEVICE)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(params=agent.parameters(), lr=LEARN_RATE)
replay = ReplayBuffer(capacity=REPLAY_SIZE)
writer = SummaryWriter(f'runs/DQN env=Pong {now()}', flush_secs=2)


# Initial replay buffer fill
print(f"Initial filling replay buffer:")
while replay.size < REPLAY_SIZE_START:
    episode = play_episode(env, agent.policy)
    replay.add(episode)
    print(f"Replay buffer size: {replay.size}/{replay.capacity}")


print(f"Training.. Target network: {TARGET_NET_ENABLED}")
exp_iterator = play_steps(env, policy=agent.policy)
mov_loss = 0
steps = 0
ts = time.time()
while True:
    steps += 1
    agent.eps = torch.tensor(max(EPS['FINAL'], EPS['INITIAL'] - steps / EPS['DURATION']))  # linear scheduler

    # collect new experience
    obs, action, reward, obs_next, terminated, truncated = next(exp_iterator)
    replay.add_step(obs, action, reward, obs_next, terminated, truncated)

    # sample batched experiences from the replay buffer
    obs, actions, rewards, obs_next, done = replay.sample(batch_size=BATCH_SIZE, device=DEVICE)

    # compute future rewards
    with torch.no_grad(): # bootstrap
        if TARGET_NET_ENABLED:
            Q_next = agent.target_net(obs_next)
        else:
            Q_next = agent(obs_next)
    R = rewards + (1 - done) * GAMMA * Q_next.max(dim=-1, keepdim=True)[0]

    # update the model
    optimizer.zero_grad()
    Q_action = agent(obs).gather(dim=-1, index=actions)
    loss = loss_fn(Q_action, R.detach())
    loss.backward()
    optimizer.step()
    mov_loss = (.9 * mov_loss + .1 * loss.item()) if steps > 1 else loss.item()

    # sync target network
    if TARGET_NET_ENABLED and steps % TARGET_NET_SYNC_STEPS == 0:
        agent.sync_target_net()


    # Collect some stats
    if steps % LOG_STEP == 0:
        n = LOG_STEP
        fps = n / (time.time() - ts)
        print(f"#{steps:>4} | {loss=:.6f}, {mov_loss=:.6f}, eps={agent.eps:.4f} | Replay buffer: avg_score={replay.stats['avg_score']=:.2f}, best_score={replay.stats['best_score']=:.2f}, avg_episode_length={replay.stats['avg_episode_length']=:.2f} | {fps=:.2f} ")
        writer.add_scalar('Replay buffer/Avg score', replay.stats['avg_score'], steps)
        writer.add_scalar('Replay buffer/Avg episode length', replay.stats['avg_episode_length'], steps)
        writer.add_scalar('Replay buffer/Best score', replay.stats['best_score'], steps)
        writer.add_scalar('Train/Mov loss', mov_loss, steps)
        writer.add_scalar('Train/FPS', fps, steps)
        writer.add_scalar('Train/Epsilon', agent.eps, steps)
        if steps % (LOG_STEP*10)  == 0:
            writer.add_histogram('hist/Rewards', rewards, steps)
            writer.add_histogram('hist/Observations', obs, steps)
            writer.add_histogram('hist/Returns', R, steps)
            writer_add_params(writer, agent.net, steps) # without the target_net
        if steps == 1000 or steps % (LOG_STEP*100) == 0:
            print(f"Evaluating the agent over {EVAL_NUM_EPISODES} episodes..")
            agent.eps = torch.tensor(EPS['FINAL'])
            score, episode_length, value = evaluate_policy_agent(env, agent, EVAL_NUM_EPISODES, DEVICE)
            print(f"Evaluation: {score=}, {episode_length=}, {value=}")
            writer.add_scalar('Eval/Avg score', score, steps)
            writer.add_scalar('Eval/Avg episode length', episode_length, steps)
            writer.add_scalar('Eval/Avg maxQ value', value, steps)
            if score >= ENV_GOAL:
                print(f"Environment solved in {steps} steps!")
                torch.save({
                    'steps': steps,
                    'model': agent.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, f'./runs/DQN-Pong - {score=:.2f}, {steps=} - {now()}.pt')
                break

        ts = time.time()

# Play one episode with visualization
print(f"Playing one episode with the trained agent")
env_ = gym.make(**ENV_SETTINGS, render_mode='human')
episode = play_episode(env_, lambda s : agent.policy(s, greedy=True))
print(f"Episode finished with total reward {episode.total_rewards}")

