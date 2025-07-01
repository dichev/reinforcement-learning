import cv2
import gymnasium as gym
import ale_py
import numpy as np
import random
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


ENVS = ('ALE/Breakout-v5', 'ALE/AirRaid-v5', 'ALE/Pong-v5')
LATENT_SIZE = 128
FILTERS_SIZE_MPLR = 64
IMAGE_SIZE = 64
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
REPORT_EVERY_ITER = 100
SAVE_IMAGE_EVERY_ITER = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class Generator(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        # pipe deconvolves input vector into (3, 64, 64) image
        self.pipe = nn.Sequential(
            nn.ConvTranspose2d(in_channels=LATENT_SIZE, out_channels=FILTERS_SIZE_MPLR * 8, kernel_size=4, stride=1, padding=0),
            nn.BatchNorm2d(FILTERS_SIZE_MPLR * 8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS_SIZE_MPLR * 8, out_channels=FILTERS_SIZE_MPLR * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS_SIZE_MPLR * 4),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS_SIZE_MPLR * 4, out_channels=FILTERS_SIZE_MPLR * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS_SIZE_MPLR * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS_SIZE_MPLR * 2, out_channels=FILTERS_SIZE_MPLR, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS_SIZE_MPLR),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=FILTERS_SIZE_MPLR, out_channels=output_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.pipe(x)


class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        # this pipe converges image into the single number
        self.conv_pipe = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=FILTERS_SIZE_MPLR, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_SIZE_MPLR, out_channels=FILTERS_SIZE_MPLR * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS_SIZE_MPLR * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_SIZE_MPLR * 2, out_channels=FILTERS_SIZE_MPLR * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS_SIZE_MPLR * 4),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_SIZE_MPLR * 4, out_channels=FILTERS_SIZE_MPLR * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(FILTERS_SIZE_MPLR * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=FILTERS_SIZE_MPLR * 8, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        conv_out = self.conv_pipe(x)
        return conv_out.flatten()



class AtariPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(low=-1., high=1., shape=(3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    def observation(self, ob):
        assert np.mean(ob) > .01

        ob = cv2.resize(ob, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        ob = np.moveaxis(ob, 2, 0).astype(np.float32)
        ob = ob * 2. / 255. - 1.  # normalize [-1, 1]
        return ob


def batches(envs, batch_size):
    for env in envs:
        env.reset()
    while True:
        batch = []
        for _ in range(batch_size):
            env = random.choice(envs)
            action = env.action_space.sample()
            ob, reward, terminated, truncated, info = env.step(action)
            batch.append(ob)
            if terminated or truncated:
                env.reset()

        batch = torch.tensor(np.stack(batch))
        yield batch



writer = SummaryWriter()
envs = [AtariPreprocessing(gym.make(name)) for name in ENVS] #, render_mode="human"))
generator = Generator(output_shape=(3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
discriminator = Discriminator(input_shape=(3, IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
loss = nn.BCELoss()
G_optimizer = torch.optim.Adam(params=generator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(params=discriminator.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))


history = []
for i, batch in enumerate(batches(envs, BATCH_SIZE)):
    X = batch.to(DEVICE)
    y_true = torch.ones(BATCH_SIZE).to(DEVICE)
    y_fake = torch.zeros(BATCH_SIZE).to(DEVICE)
    z = torch.randn(BATCH_SIZE, LATENT_SIZE, 1, 1).to(DEVICE)


    # Train Discriminator
    D_optimizer.zero_grad()
    X_fake = generator(z)
    y_hat_true = discriminator(X)
    y_hat_fake = discriminator(X_fake.detach()) # generator is detached
    D_loss = loss(y_hat_true, y_true) + loss(y_hat_fake, y_fake)
    D_loss.backward()
    D_optimizer.step()

    # Train Generator
    G_optimizer.zero_grad()
    y_hat = discriminator(X_fake)
    G_loss = loss(y_hat, y_true)
    G_loss.backward()
    G_optimizer.step()

    history.append((D_loss.detach(), G_loss.detach()))

    if i % REPORT_EVERY_ITER == 0:
        print(f"#{i:>5}. Losses: discriminator: {D_loss.item():.6f} generator: {G_loss.item():.6f}")
        D_losses, G_losses = zip(*history)
        writer.add_scalars("losses", {
            "gen_loss": torch.stack(G_losses).mean(),
            "dis_loss": torch.stack(D_losses).mean()
        }, i)
        history = []

    if i % SAVE_IMAGE_EVERY_ITER == 0:
        writer.add_image("fake", make_grid(X_fake.data, normalize=True), i)
        writer.add_image("real", make_grid(batch.data, normalize=True), i)
