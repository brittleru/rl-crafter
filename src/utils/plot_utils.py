import os.path

import matplotlib.pyplot as plt

from src.utils.constant_builder import PathBuilder


def plot_epsilon(steps, epsilons, network_type: str, is_grid: bool = True):
    plt.figure()
    plt.plot(steps, epsilons, color="blueviolet")
    plt.title(f"ε drop over the steps for {network_type}")
    plt.legend(["ε"])
    plt.xlabel("Step")
    plt.ylabel("Epsilon values")
    plt.grid(visible=is_grid)
    plt.savefig(os.path.join(PathBuilder.TRAIN_PLOTS_DIR, f"{network_type.lower()}_epsilon.png"))
    # plt.show()


def plot_rewards(steps, scores, network_type: str, is_grid: bool = True):
    plt.figure()
    plt.plot(steps, scores, color="green")
    plt.title(f"Scores over the steps for {network_type}")
    plt.legend(["scores"])
    plt.xlabel("Step")
    plt.ylabel("Score value")
    plt.grid(visible=is_grid)
    plt.savefig(os.path.join(PathBuilder.TRAIN_PLOTS_DIR, f"{network_type.lower()}_scores.png"))
    # plt.show()
