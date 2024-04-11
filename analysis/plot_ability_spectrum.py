import pathlib
from typing import List, Dict, Tuple

import numpy as np
from matplotlib import pyplot as plt

from src.utils.computing import compute_success_rates
from src.utils.plot_json_utils import compile_all_agents_seeds


def plot_all_agents_spectrum(
        agent_dirs: List[str], save_path: str, legend: Dict[str, str],
        colors: List[str], clip_seeds_to: int = 3, sort=False, fig_size: Tuple[int, int] = (7, 3)
):
    t_runs = compile_all_agents_seeds(agent_dirs=agent_dirs, clip_seeds_to=clip_seeds_to)
    percents, algorithms, seeds, tasks = compute_success_rates(
        t_runs, sort_by=sort and legend and list(legend.keys())[0])
    tasks = tasks[0][0]
    if not legend:
        algorithms = sorted(set(run['algorithm'] for run in t_runs))
        legend = {x: x.replace('_', ' ').title() for x in algorithms}

    fig, ax = plt.subplots(figsize=fig_size)
    centers = np.arange(len(tasks))
    width = 0.7

    for index, (algorithm, label) in enumerate(legend.items()):
        heights = np.nanmean(percents[algorithms.index(algorithm)], 0)
        pos = centers + width * (0.5 / len(algorithms) + index / len(algorithms) - 0.5)
        color = colors[index]
        ax.bar(pos, heights[index, 0, :], width / len(algorithms), label=label, color=color)

    names = [x[len('achievement_'):].replace('_', ' ').title() for x in tasks]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(
        axis='x', which='both', width=14, length=0.8, direction='inout')
    ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
    ax.set_xticks(centers + 0.0)
    ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor')

    ax.set_ylabel('Success Rate (%)')
    ax.set_yscale('log')
    ax.set_ylim(0.01, 100)
    ax.set_yticks([0.01, 0.1, 1, 10, 100])
    ax.set_yticklabels('0.01 0.1 1 10 100'.split())

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.legend(
        loc='upper center', ncol=10, frameon=False, borderpad=0, borderaxespad=0)

    pathlib.Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path)
    print(f'Saved {save_path}')


if __name__ == '__main__':
    import os
    from src.utils.constant_builder import PathBuilder, AgentTypes

    clip_to_num_seeds = 4
    in_dirs = [
        PathBuilder.RANDOM_AGENT_LOG_DIR,
        PathBuilder.DQN_AGENT_LOG_DIR,
        PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR,
    ]

    agent_legend = {
        AgentTypes.RANDOM: "Random",
        AgentTypes.DQN: "DQN",
        AgentTypes.DDQN: "DDQN",
    }
    agent_colors = [
        "#377eb8", "#5fc35d", "#984ea3",
    ]
    path_to_save = os.path.join(PathBuilder.SCORE_PLOTS_DIR, "agents_spectrum_rand_dqn_ddqn.png")
    plot_all_agents_spectrum(
        agent_dirs=in_dirs,
        save_path=path_to_save,
        legend=agent_legend,
        colors=agent_colors,
        clip_seeds_to=clip_to_num_seeds
    )

    in_dirs = [
        PathBuilder.DUELING_DQN_AGENT_LOG_DIR,
        PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR,
        PathBuilder.RAINBOW_DQN_AGENT_LOG_DIR,
    ]

    agent_legend = {
        AgentTypes.DUELING_DQN: "Duel DQN",
        AgentTypes.DUELING_DOUBLE_DQN: "Duel DDQN",
        AgentTypes.RAINBOW: "Rainbow",
    }
    agent_colors = [
        "#bf3217", "#de9f42", "#6a554d"
    ]
    path_to_save = os.path.join(PathBuilder.SCORE_PLOTS_DIR, "agents_spectrum_dueldqn_duelddqn_rainbow.png")
    plot_all_agents_spectrum(
        agent_dirs=in_dirs,
        save_path=path_to_save,
        legend=agent_legend,
        colors=agent_colors,
        clip_seeds_to=clip_to_num_seeds
    )

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
    agent_colors = [
        "#377eb8", "#5fc35d", "#984ea3",
        "#bf3217", "#de9f42", "#6a554d"

    ]
    path_to_save = os.path.join(PathBuilder.SCORE_PLOTS_DIR, "agents_spectrum_all.png")
    plot_all_agents_spectrum(
        agent_dirs=in_dirs,
        save_path=path_to_save,
        legend=agent_legend,
        colors=agent_colors,
        clip_seeds_to=clip_to_num_seeds,
        fig_size=(14, 6)
    )
