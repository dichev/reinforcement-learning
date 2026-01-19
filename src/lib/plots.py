import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx


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



def plot_monte_carlo_search_tree(root, label_actions=True, title='Monte Carlo Search Tree'):
    G = nx.DiGraph()
    pos = {}

    # Built a graph and compute a circular layout
    def traverse(node, start=0.0, end=2 * np.pi, depth=0):
        n_id = id(node)
        angle = (start + end) / 2
        pos[n_id] = (depth * np.cos(angle), depth * np.sin(angle))
        G.add_node(n_id, visits=node.visits, value=node.scores)
        if node.children:
            span = (end - start) / len(node.children)
            for i, (action, child) in enumerate(node.children.items()):
                G.add_edge(n_id, id(child), action=action)
                traverse(child, start + i * span, start + (i + 1) * span, depth + 1)

    traverse(root)
    node_list = list(G.nodes())


    # Plot
    fig, ax = plt.subplots(figsize=(14, 12))
    ax.set_aspect('equal')
    ax.axis('off')


    # Draw Nodes
    values = np.array([G.nodes[n]['value'] for n in node_list])
    limit = max(abs(values[1:].min()), abs(values[1:].max()), 1.0) # skip the root as it is an outlier
    nodes = nx.draw_networkx_nodes(G, pos, nodelist=node_list, node_size=100, node_color=values, cmap='RdYlGn', vmin=-limit, vmax=limit, edgecolors='black', linewidths=.3, ax=ax)


    # Draw Edges
    visits = np.array([G.nodes[n]['visits'] for n in node_list])
    norm_v = np.sqrt(visits / visits[1:].max())
    for u, v in G.edges():
        idx = node_list.index(v)
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=0.1 + 2.0 * norm_v[idx], alpha=0.2 + 0.6 * norm_v[idx], edge_color="black", ax=ax, arrowsize=8)


    # Draw action labels
    if label_actions:
        edge_labels = nx.get_edge_attributes(G, 'action')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, rotate=False)


    # Legend
    cbar = plt.colorbar(nodes, ax=ax, orientation='horizontal', pad=0.08, shrink=0.5)
    cbar.set_label('Node Value (Score)', fontsize=10)
    legend_elements = [
        Line2D([0], [0], color='#2c3e50', lw=6, label=f'High Visits ({int(visits.max())})'),
        Line2D([0], [0], color='#2c3e50', lw=1, label='Low Visits (1)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False)

    if title:
        plt.title(title)

    plt.tight_layout()
    plt.show()




if __name__ == '__main__':
    print_binary_tree_array(list(range(2**5 - 4)))
    print_binary_tree_array((np.arange(2**5 - 4) + .111).tolist())
