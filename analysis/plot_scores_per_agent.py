import pathlib
from typing import List, Dict

import numpy as np
from matplotlib import pyplot as plt

from src.utils.computing import compute_success_rates, compute_scores
from src.utils.plot_json_utils import compile_all_agents_seeds


def plot_all_agent_scores(
        agent_dirs: List[str], save_path: str, legend: Dict[str, str], colors: List[str],
        y_lim: float = None, clip_seeds_to: int = 3
):
    t_runs = compile_all_agents_seeds(agent_dirs=agent_dirs, clip_seeds_to=clip_seeds_to)

    percents, algorithms, seeds, tasks = compute_success_rates(runs=t_runs)
    scores = compute_scores(percents)
    if not legend:
        algorithms = sorted(set(run["method"] for run in t_runs))
        legend = {x: x.replace("_", " ").title() for x in algorithms}
    legend = dict(reversed(legend.items()))

    scores = scores[np.array([algorithms.index(m) for m in legend.keys()])]
    mean = np.nanmean(scores, -1)
    std = np.nanstd(scores, -1)

    fig, ax = plt.subplots(figsize=(4, 3))
    centers = np.arange(len(legend))
    width = 0.7
    colors = list(reversed(colors[:len(legend)]))
    error_kw = dict(capsize=5, c="#000")
    ax.bar(centers, mean, yerr=std, color=colors, error_kw=error_kw)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(
        axis="x", which="both", width=50, length=0.8, direction="inout")
    ax.set_xlim(centers[0] - 2 * (1 - width), centers[-1] + 2 * (1 - width))
    ax.set_xticks(centers + 0.0)
    ax.set_xticklabels(
        list(legend.values()), rotation=45, ha="right", rotation_mode="anchor")

    ax.set_ylabel("Crafter Score (%)")
    if y_lim:
        ax.set_ylim(0, y_lim)

    fig.tight_layout()
    pathlib.Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(save_path)
    print(f"Saved {save_path}")


if __name__ == '__main__':
    import os
    from src.utils.constant_builder import PathBuilder, AgentTypes

    clip_to_num_seeds = 4
    in_dirs = [
        PathBuilder.RANDOM_AGENT_LOG_DIR,
        PathBuilder.DQN_AGENT_LOG_DIR,
        PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR,
        PathBuilder.DUELING_DQN_AGENT_LOG_DIR,
        PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR,
        PathBuilder.RAINBOW_DQN_AGENT_LOG_DIR,
    ]

    agent_legend = {
        AgentTypes.RAINBOW: "Rainbow",
        AgentTypes.DUELING_DOUBLE_DQN: "Duel DDQN",
        AgentTypes.DUELING_DQN: "Duel DQN",
        AgentTypes.DDQN: "DDQN",
        AgentTypes.DQN: "DQN",
        AgentTypes.RANDOM: "Random",
    }
    agent_colors = ["#377eb8", "#5fc35d", "#984ea3", "#bf3217", "#de9f42", "#6a554d"]
    path_to_save = os.path.join(PathBuilder.SCORE_PLOTS_DIR, "agents_scores.png")
    plot_all_agent_scores(
        agent_dirs=in_dirs,
        save_path=path_to_save,
        legend=agent_legend,
        colors=agent_colors,
        y_lim=4,
        clip_seeds_to=clip_to_num_seeds
    )
