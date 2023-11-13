import os.path

import matplotlib.pyplot as plt

from src.utils.constant_builder import PathBuilder
from src.utils.file_utils import create_dir_if_doesnt_exist


def plot_epsilon(steps, epsilons, network_type: str, model_num: str = "99", is_grid: bool = True):
    plt.figure()
    plt.plot(steps, epsilons, color="blueviolet")
    plt.title(f"ε drop over the steps for {network_type}")
    plt.legend(["ε"])
    plt.xlabel("Step")
    plt.ylabel("Epsilon values")
    plt.grid(visible=is_grid)
    plot_path = os.path.join(PathBuilder.TRAIN_PLOTS_DIR, model_num)
    create_dir_if_doesnt_exist(plot_path)
    plt.savefig(os.path.join(plot_path, f"{network_type.lower()}_epsilon.png"))


def plot_rewards(steps, scores, network_type: str, model_num: str = "99", is_grid: bool = True):
    plt.figure()
    plt.plot(steps, scores, color="green")
    plt.title(f"Scores over the steps for {network_type}")
    plt.legend(["scores"])
    plt.xlabel("Step")
    plt.ylabel("Score value")
    plt.grid(visible=is_grid)
    plot_path = os.path.join(PathBuilder.TRAIN_PLOTS_DIR, model_num)
    create_dir_if_doesnt_exist(plot_path)
    plt.savefig(os.path.join(plot_path, f"{network_type.lower()}_scores.png"))
