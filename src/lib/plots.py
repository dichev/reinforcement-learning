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