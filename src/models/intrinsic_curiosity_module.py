import torch, torch.nn as nn, torch.nn.functional as F


class IntrinsicCuriosityModule(nn.Module):
    """
    Curiosity-driven Exploration by Self-supervised Prediction
    Paper: https://arxiv.org/pdf/1705.05363
    """

    def __init__(self, n_actions, n_frames, emb_size=512):
        super().__init__()
        self.state_encoder = StateEncoder(n_frames, emb_size)
        self.inverse_predictor = InversePredictor(n_actions, emb_size)
        self.forward_predictor = ForwardPredictor(n_actions, emb_size)
        self.n_actions = n_actions

    def forward(self, state, state_next, action):
        phi1 = self.state_encoder(state)
        phi2 = self.state_encoder(state_next)

        a_pred = self.inverse_predictor(phi1, phi2)
        phi2_pred = self.forward_predictor(phi1.detach(), action)

        phi_diff = phi2_pred - phi2.detach()
        return a_pred, phi_diff

    def predict(self, state, state_next, actions, beta=.2, eta=1.):
        a_onehot = F.one_hot(actions, self.n_actions).float()
        a_pred, phi_diff = self(state, state_next, a_onehot)

        inverse_loss = F.cross_entropy(a_pred, actions)
        forward_loss = .5 * (phi_diff ** 2).mean()  # MSE/2
        total_loss = (1 - beta) * inverse_loss + beta * forward_loss
        intrinsic_rewards = .5 * eta * (phi_diff.detach() ** 2).sum(dim=-1)  # or reuse the forward loss

        return intrinsic_rewards.detach(), total_loss, { 'inverse_loss': inverse_loss.detach(), 'forward_loss': forward_loss.detach() }


class StateEncoder(nn.Module):
    def __init__(self, n_frames, emb_size):
        super().__init__()
        self.net = nn.Sequential(                                                         # if:  n, 84, 84
            nn.Conv2d(in_channels=n_frames, out_channels=32, kernel_size=8, stride=4),    # ->  32, 20, 20
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),          # ->  64,  9,  9
            nn.ELU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),          # ->  64,  7,  7
            nn.ELU(),
            nn.Flatten(),                                                                 # -> 3136 (flatten)
            nn.LazyLinear(emb_size),                                                      # -> 512
            nn.ELU(), # hmm
        )

    def forward(self, state): # S -> E
        B, C, H, W = state.shape
        emb = self.net(state)
        return emb


class InversePredictor(nn.Module):
    def __init__(self, n_actions, emb_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=2*emb_size, out_features=emb_size//2),
            nn.ReLU(),
            nn.Linear(in_features=emb_size//2, out_features=n_actions),
        )

    def forward(self, state_emb, state_emb_next): # S x S' -> ~a
        state = torch.cat((state_emb, state_emb_next), dim=-1)
        B, E = state.shape
        a_logits = self.net(state)
        return a_logits


class ForwardPredictor(nn.Module):
    def __init__(self, n_actions, emb_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=emb_size + n_actions, out_features=emb_size//2),
            nn.ReLU(),
            nn.Linear(in_features=emb_size//2, out_features=emb_size),
        )

    def forward(self, state_emb, actions_onehot):  # S x a -> S'
        assert len(state_emb) == len(actions_onehot)
        assert state_emb.grad_fn is None, "The embedded state was not detached"
        B, E = state_emb.shape
        B, A = actions_onehot.shape

        x = torch.cat((state_emb, actions_onehot), dim=-1)
        x = self.net(x)
        return x



if __name__ == "__main__":
    # Test data and settings
    n_actions = 12
    B, C, H, W = 10, 4, 84, 84
    obs = torch.randn(B, C, H, W)
    obs_next = torch.randn(B, C, H, W)
    actions = torch.randint(n_actions, size=(B,))

    # ICM module
    icm = IntrinsicCuriosityModule(n_actions, n_frames=C)
    print(icm)
    intrinsic_rewards, loss, stats = icm.predict(obs, obs_next, actions)
    inverse_loss, forward_loss = stats['inverse_loss'], stats['forward_loss']
    print(f'{intrinsic_rewards=}\n{loss=:f}\n{inverse_loss=:f}\n{forward_loss=:f}\n')
