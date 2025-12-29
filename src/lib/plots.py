import math
import numpy as np
from matplotlib import pyplot as plt

def plot_value_distribution(ob, support, probs, labels, title):
    ob, support, probs = ob.detach().cpu(), support.detach().cpu(), probs.detach().cpu()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))
    ax1.imshow(ob.permute(1, 2, 0))
    ax1.axis('off')
    ax1.set_title(title)

    for p, label in zip(probs, labels):
        ax2.bar(support, p, alpha=0.5, width=0.4, label=label)
    ax2.legend(loc='lower right', fontsize='small')
    ax2.yaxis.set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    plt.tight_layout()
    return fig



def print_binary_tree_array(tree, max_precision=2, empty_node="•"):
    if not tree: return
    height = math.ceil(math.log2(len(tree) + 1))
    total_nodes = 2 ** height

    # Format values and compute max width
    value_width = 2
    nodes_formatted = []
    for i in range(total_nodes):
        v = tree[i] if i < len(tree) else None  # we will loop also through missing nodes
        v = f" {v:.{max_precision}f} " if isinstance(v, float) else f'{v} ' if v is not None else empty_node
        nodes_formatted.append(v)
        value_width = max(value_width, len(v))
    if value_width % 2 == 1:  # even number centers better
        value_width += 1

    for level in range(height):
        # Print node values
        n = 2 ** level  # total nodes on the level
        nodes = nodes_formatted[n-1 : 2*n-1]
        slot_width = 2 ** (height - level - 1) * value_width
        print("".join(v.center(slot_width) for v in nodes))

        # Print the edges
        if level < height - 1:
            arm = "─" * (slot_width // 4 - 1)
            connector = f"┌{arm}┴{arm}┐".center(slot_width)
            edge_line = [connector for v in nodes]
            print("".join(edge_line))


def draw_policy_grid(Q, grid, action_map, grid_map, title=''):
    if title: print(title)

    rows, cols = len(grid), len(grid[0])
    policy = np.where(Q.min(axis=1) == Q.max(axis=1), -1, Q.argmax(axis=1))
    policy_grid = policy.reshape(rows, cols)
    for i in range(rows):
        line = [action_map[action] if action != -1 else grid_map[cell]
                for cell, action in zip(grid[i], policy_grid[i])]
        print(' '.join(line))


if __name__ == '__main__':
    print_binary_tree_array(list(range(2**5 - 4)))
    print_binary_tree_array((np.arange(2**5 - 4) + .111).tolist())
