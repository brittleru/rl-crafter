import pathlib
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from src.utils.computing import compute_success_rates, binning
from src.utils.plot_json_utils import compile_all_agents_seeds


def plot_counts_one_agent(
        agent_dir: str, save_path: str, color, cols=4, size=(2, 1.8), clip_seeds_to: int = 3, agent_steps: int = 100_000
):
    t_runs = compile_all_agents_seeds(agent_dirs=[agent_dir], clip_seeds_to=clip_seeds_to, agent_steps=agent_steps)
    percents, algorithms, seeds, tasks = compute_success_rates(runs=t_runs)
    borders = np.arange(0, agent_steps, 1e4)
    keys = ['reward', 'length'] + tasks

    rows = len(keys) // cols
    fig, axes = plt.subplots(
        nrows=rows, ncols=cols, figsize=(size[0] * cols, size[1] * rows))

    for ax, key in zip(axes.flatten(), keys):
        ax.set_title(key.replace('achievement_', '').replace('_', ' ').title())
        ys = np.concatenate([run[key] for run in t_runs])
        xs = np.concatenate([run['xs'] for run in t_runs])

        binxs, binys = binning(xs, ys, borders, np.nanmean)
        ax.plot(binxs, binys, color=color)

        mins = binning(xs, ys, borders, np.nanmin)[1]
        maxs = binning(xs, ys, borders, np.nanmax)[1]
        ax.fill_between(binxs, mins, maxs, linewidths=0, alpha=0.2, color=color)

        ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5, steps=[1, 2, 2.5, 5, 10]))

        if maxs.max() == 0:
            ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    pathlib.Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path)
    print(f'Saved {save_path}')


if __name__ == '__main__':
    import os
    from src.utils.constant_builder import PathBuilder, AgentTypes

    in_dirs = [
        PathBuilder.RANDOM_AGENT_LOG_DIR,
        PathBuilder.DQN_AGENT_LOG_DIR,
        PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR,
        PathBuilder.DUELING_DQN_AGENT_LOG_DIR,
        PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR,
        PathBuilder.RAINBOW_DQN_AGENT_LOG_DIR,
    ]

    agent_legend = {
        AgentTypes.RANDOM: "Random",
        AgentTypes.DQN: "DQN",
        AgentTypes.DDQN: "DDQN",
        AgentTypes.DUELING_DQN: "Duel DQN",
        AgentTypes.DUELING_DOUBLE_DQN: "Duel DDQN",
        AgentTypes.RAINBOW: "Rainbow",
    }
    agent_colors = ["#377eb8", "#5fc35d", "#984ea3", "#bf3217", "#de9f42", "#6a554d"]

    for agent_dir, agent_name, agent_color in zip(in_dirs, agent_legend, agent_colors):
        path_to_save = os.path.join(PathBuilder.ACHIEVEMENT_PLOTS_DIR, f"counts_{agent_name.lower()}.png")

        plot_counts_one_agent(
            agent_dir=agent_dir,
            save_path=path_to_save,
            color=agent_color,
            clip_seeds_to=3
        )