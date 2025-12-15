import torch
from torch import nn


class RelationalBlock(nn.Module): # almost the same as the transformer block:
    """
    Paper: Deep reinforcement learning with relational inductive biases
    https://openreview.net/pdf?id=HkxaFoC9KQ
    """

    def __init__(self, emb_size, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.attention = nn.MultiheadAttention(emb_size, num_heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(emb_size, emb_size*4),
            nn.ReLU(),
            nn.Linear(emb_size*4, emb_size),
        )
        self.norm2 = nn.LayerNorm(emb_size)

    def forward(self, x): # E = LN(E+f(attention(LN(E))
        x_skip = x
        x = self.norm1(x)
        x, attn_weights = self.attention(x, x, x)
        x = x_skip + self.ff(x) # note there is a single skip connection
        x = self.norm2(x)
        return x  # B, L, E


class RelationalModel(nn.Module): # following the described architecture for Box-World
    """
    Paper: Deep reinforcement learning with relational inductive biases
    https://openreview.net/pdf?id=HkxaFoC9KQ
    """
    def __init__(self, num_actions, input_shape=(3, 12, 12), emb_size=64, num_heads=4, num_shared_blocks=2, with_value_head=True):
        super().__init__()
        self.input_shape = input_shape
        self.num_shared_blocks = num_shared_blocks

        # Conv stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_shape[0],  12, kernel_size=2), nn.ReLU(), #nn.MaxPool2d(2),
            nn.Conv2d(12, 24, kernel_size=2), nn.ReLU(), #nn.MaxPool2d(2),
        )

        # Relational module
        self.register_buffer('coords', self._generate_coords())
        self.proj = nn.Linear(24+2, emb_size) # channels + 2 positional encodings -> emb_size
        self.relational_block = RelationalBlock(emb_size, num_heads)

        # Fully connected layers - to process relational information into a policy
        self.fc = nn.Sequential(
            nn.Linear(emb_size, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, emb_size), nn.ReLU(),
        )
        self.policy_head = nn.Linear(emb_size, num_actions)
        self.value_head = nn.Linear(emb_size, 1) if with_value_head else None

    def forward(self, x):
        x = self.stem(x)
        B, C, H, W = x.shape  # note original H, W were downsampled by the conv layers

        x = x.permute(0, 2, 3, 1).view(B, H*W, C)                           # B, C, H, W -> B, HW, C = B, L, E
        pos_encodings = self.coords.expand(B, -1, -1).to(x.device)          # B, L, 2
        x = torch.cat((x, pos_encodings), dim=-1)                   # B, L, C+2
        x = self.proj(x)                                                    # B, L, E

        for _ in range(self.num_shared_blocks):
            x = self.relational_block(x)  # B, L, E

        x = x.amax(dim=1)   # B, E - max-pooling over the entity dimension
        x = self.fc(x)      # B, E
        logits = self.policy_head(x)
        if not self.value_head:
            return logits

        value = self.value_head(x)
        return logits, value

    @torch.no_grad()
    def conv_out_shape(self, device=None, batch_size=1):
        dummy = torch.zeros(batch_size, *self.input_shape).to(device)
        B, C, H, W = self.stem(dummy).shape
        if H > 10 or W > 10: print(f'! Warning: conv stem output size ({H}x{W}) might be too large, consider downsampling')
        return B, C, H, W

    def _generate_coords(self):
        _, _, H, W = self.conv_out_shape()
        pos_y, pos_x = torch.meshgrid(torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij")
        pos_xy = torch.stack((pos_x, pos_y), dim=-1)
        pos_xy = pos_xy.reshape(1, H * W, 2)
        return pos_xy


if __name__ == "__main__":
    B, C, H, W = 32, 3, 12, 12
    model = RelationalModel(num_actions=4, input_shape=(C, H, W))
    x = torch.randn(B, C, H, W)
    logits, value = model(x)
    print(model)
    print(sum(p.numel() for p in model.parameters()), 'parameters')
    print(f'input obs: {tuple(x.shape)}')
    print('conv out:', model.conv_out_shape(batch_size=B))
    print(f'output: logits={tuple(logits.shape)} | value={tuple(value.shape)}')
