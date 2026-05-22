import numpy as np
import matplotlib.pyplot as plt

def compare_values(
    expls,
    labels=None,
    colors=None,
    idx=0,
    figsize=(9, 5.5)
):
    n_expls = len(expls)

    # Defaults
    if labels is None:
        labels = [f"expl_{i}" for i in range(n_expls)]

    if colors is None:
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i % 10) for i in range(n_expls)]

    # Feature info
    num_features = expls[0].values.shape[1]
    feature_names = expls[0].feature_names

    # Figure
    plt.figure(figsize=figsize)

    # Dynamic bar width
    total_width = 0.8
    bar_width = total_width / n_expls

    x = np.arange(num_features)

    # Center bars around feature position
    offsets = (
        np.arange(n_expls) - (n_expls - 1) / 2
    ) * bar_width

    # Plot bars
    for i, expl in enumerate(expls):
        plt.bar(
            x + offsets[i],
            expl.values[idx],
            width=bar_width,
            label=labels[i],
            color=colors[i]
        )

    # Formatting
    plt.legend(fontsize=16)
    plt.tick_params(labelsize=14)
    plt.ylabel("SHAP Values", fontsize=16)
    plt.title("Census Explanation Example", fontsize=18)

    plt.xticks(
        x,
        feature_names,
        rotation=35,
        rotation_mode="anchor",
        ha="right"
    )

    plt.tight_layout()
    plt.show()