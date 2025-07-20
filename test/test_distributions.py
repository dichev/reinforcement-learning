import torch
import pytest

from lib.calc import categorical_projection



def test_categorical_projection(plot=True):
    probs_fixed = torch.tensor([.05, .07, .09, .10, .11, .16, .11, .10, .09, .07, .05])
    support = torch.linspace(0, len(probs_fixed) - 1, len(probs_fixed))
    v_min, v_max = support.min(), support.max()

    def case(rewards, gamma, dones, expected):
        expected = torch.as_tensor(expected, dtype=torch.float)
        B, N_atoms = expected.shape
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)
        probs = probs_fixed.expand(B, N_atoms)

        # Bellman update operator (over the atoms):
        Tz = rewards + (1 - dones) * gamma * support

        # Projection onto the fixed support
        proj = categorical_projection(Tz, probs, v_min, v_max)

        # plot and testing
        if plot:
            from matplotlib import pyplot as plt
            for source_, target, reward in zip(probs, proj, rewards):
                plt.bar(support, source_, alpha=0.6, width=0.35, label="source")
                plt.bar(support, target, alpha=0.6, width=0.35, label="projected")
                plt.axvline(reward, label=f'{reward=}', color='black')
                plt.title(f'{reward=}, {gamma=}')
                plt.legend()
                plt.show()
        torch.testing.assert_close(proj.sum(dim=-1), torch.ones(B))  # valid distribution sums to 1
        torch.testing.assert_close(proj, expected)

    # Collapse
    case(rewards=[[7.00]], gamma=0., dones=[[0]], expected=[[0, 0, 0, 0, 0, 0, 0,   1,   0, 0, 0]])
    case(rewards=[[7.50]], gamma=0., dones=[[0]], expected=[[0, 0, 0, 0, 0, 0, 0, .50, .50, 0, 0]])
    case(rewards=[[7.75]], gamma=0., dones=[[0]], expected=[[0, 0, 0, 0, 0, 0, 0, .25, .75, 0, 0]])

    # Edge & outside support
    case(rewards=[[0.00]], gamma=0., dones=[[0]], expected=[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    case(rewards=[[10.0]], gamma=0., dones=[[0]], expected=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    case(rewards=[[20.0]], gamma=0., dones=[[0]], expected=[[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    case(rewards=[[-5.0]], gamma=0., dones=[[0]], expected=[[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # Gamma effect
    case(rewards=[[ 0.0]], gamma=1., dones=[[0]], expected=probs_fixed.view(1, -1))
    case(rewards=[[ 1.0]], gamma=1., dones=[[0]], expected=[[0, .05, .07, .09, .10, .11, .16, .11, .10, .09, .07 + .05]])
    case(rewards=[[-1.0]], gamma=1., dones=[[0]], expected=[[.05 + .07, .09, .10, .11, .16, .11, .10, .09, .07, .05, 0]])
    case(rewards=[[ 5.0]], gamma=.5, dones=[[0]], expected=[[0, 0, 0, 0, 0, 0.085, 0.175, 0.24, 0.24, 0.175, 0.085]])
    case(rewards=[[ 7.0]], gamma=.9, dones=[[0]], expected=[[0, 0, 0, 0, 0, 0, 0, 0.057, 0.081, 0.102, 0.76]])
    case(rewards=[[ 3.0]], gamma=.5, dones=[[0]], expected=[[0, 0, 0, .085, .175, .24, .24, .175, .085, 0, 0]])
    case(rewards=[[-2.0]], gamma=.8, dones=[[0]], expected=[[.27, .128, .204, .128, .114, .092, .064, 0, 0, 0, 0]])
    case(rewards=[[ 2.0]], gamma=.2, dones=[[0]], expected=[[0, 0, .222, .556, .222, 0, 0, 0, 0, 0, 0]])

    # Batched
    case(rewards=[[7.00], [7.50], [7.75]], gamma=0., dones=[[0],[0],[0]], expected=[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, .50, .50, 0, 0], [0, 0, 0, 0, 0, 0, 0, .25, .75, 0, 0]])
    case(rewards=[[ 3.0],[ 2.1],[-1.0],[ 0.0]], gamma=.5, dones=[[0],[0],[0],[0]], expected=[[0, 0, 0, .085, .175, .24, .24, .175, .085, 0, 0], [0, 0, .073, .168, .232, .246, .18, .096, .005, 0, 0], [.26, .24, .24, .175, .085, 0, 0, 0, 0, 0, 0], [.085, .175, .24, .24, .175, .085, 0, 0, 0, 0, 0]])

    # Dones (done=1 -> gamma=0)
    case(rewards=[[ 2.0]], gamma=.1, dones=[[1]], expected=[[0, 0, 1, 0, 0, 0, 0,  0, 0, 0, 0]])
    case(rewards=[[7.00], [7.50], [7.75]], gamma=.9, dones=[[1],[1],[1]], expected=[[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, .50, .50, 0, 0], [0, 0, 0, 0, 0, 0, 0, .25, .75, 0, 0]])