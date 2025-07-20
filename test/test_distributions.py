import torch
import pytest

from lib.calc import distributional_bellman



def test_distributional_bellman(plot=True):
    source = torch.tensor([.05, .07, .09, .10, .11, .16, .11, .10, .09, .07, .05])
    support = torch.linspace(0, len(source) - 1, len(source))

    def case(reward, gamma, expected):
        target = distributional_bellman(support, source, reward, gamma)
        if plot:
            from matplotlib import pyplot as plt
            plt.bar(support, source, alpha=0.6, width=0.35, label="source")
            plt.bar(support, target, alpha=0.6, width=0.35, label="projected")
            plt.axvline(reward, label=f'{reward=}', color='black')
            plt.title(f'{reward=}, {gamma=}')
            plt.legend()
            plt.show()
        torch.testing.assert_close(target.sum(), torch.tensor(1.))  # valid distribution sums to 1
        torch.testing.assert_close(target, torch.as_tensor(expected, dtype=torch.float))

    # Collapse
    case(reward=7.00, gamma=0., expected=[0, 0, 0, 0, 0, 0, 0,   1,   0, 0, 0])
    case(reward=7.50, gamma=0., expected=[0, 0, 0, 0, 0, 0, 0, .50, .50, 0, 0])
    case(reward=7.75, gamma=0., expected=[0, 0, 0, 0, 0, 0, 0, .25, .75, 0, 0])

    # Edge & outside support
    case(reward=0.00, gamma=0., expected=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    case(reward=10.0, gamma=0., expected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    case(reward=20.0, gamma=0., expected=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    case(reward=-5.0, gamma=0., expected=[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

    # Gamma effect
    case(reward= 0.0, gamma=1., expected=source)
    case(reward= 1.0, gamma=1., expected=[0, .05, .07, .09, .10, .11, .16, .11, .10, .09, .07 + .05])
    case(reward=-1.0, gamma=1., expected=[.05 + .07, .09, .10, .11, .16, .11, .10, .09, .07, .05, 0])
    case(reward= 5.0, gamma=.5, expected=[0, 0, 0, 0, 0, 0.085, 0.175, 0.24, 0.24, 0.175, 0.085])
    case(reward= 7.0, gamma=.9, expected=[0, 0, 0, 0, 0, 0, 0, 0.057, 0.081, 0.102, 0.76])
    case(reward= 3.0, gamma=.5, expected=[0, 0, 0, .085, .175, .24, .24, .175, .085, 0, 0])
    case(reward=-2.0, gamma=.8, expected=[.27, .128, .204, .128, .114, .092, .064, 0, 0, 0, 0])
    case(reward= 2.0, gamma=.2, expected=[0, 0, .222, .556, .222, 0, 0, 0, 0, 0, 0])
