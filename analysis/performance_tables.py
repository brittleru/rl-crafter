import collections
from typing import List, Dict

import numpy as np

from src.utils.computing import compute_success_rates, compute_scores
from src.utils.plot_json_utils import compile_all_agents_seeds
from src.utils.constant_builder import PathBuilder, AgentTypes


def print_scores(agent_dirs: List[str], legend: Dict[str, str], clip_seeds_to: int = 3):
    t_runs = compile_all_agents_seeds(agent_dirs=agent_dirs, clip_seeds_to=clip_seeds_to)
    percents, methods, seeds, tasks = compute_success_rates(t_runs)
    scores = compute_scores(percents)
    if not legend:
        methods = sorted(set(run['method'] for run in t_runs))
        legend = {x: x.replace('_', ' ').title() for x in methods}

    scores = scores[np.array([methods.index(m) for m in legend.keys()])]
    means = np.nanmean(scores, -1)
    stds = np.nanstd(scores, -1)

    print('')
    print(r'\textbf{Method} & \textbf{Score} \\')
    print('')
    for method, mean, std in zip(legend.values(), means, stds):
        mean = f'{mean:.1f}'
        mean = (r'\o' if len(mean) < 4 else ' ') + mean
        print(rf'{method:<25} & ${mean} \pm {std:4.1f}\%$ \\')
    print('')


def print_spectrum(agent_dirs: List[str], legend: Dict[str, str], clip_seeds_to: int = 3, sort=False):
    t_runs = compile_all_agents_seeds(agent_dirs=agent_dirs, clip_seeds_to=clip_seeds_to)
    percents, methods, seeds, tasks = compute_success_rates(t_runs)
    scores = compute_scores(percents)
    if not legend:
        methods = sorted(set(run['method'] for run in t_runs))
        legend = {x: x.replace('_', ' ').title() for x in methods}

    scores = np.nanmean(scores, 1)
    percents = np.nanmean(percents, 1)

    if sort:
        first = next(iter(legend.keys()))
        tasks = sorted(tasks, key=lambda task: -np.nanmean(percents[first, task]))
    legend = dict(reversed(legend.items()))

    cols = ''.join(rf' & \textbf{{{k}}}' for k in legend.values())
    print(r'\newcommand{\o}{\hphantom{0}}')
    print(r'\newcommand{\b}[1]{\textbf{#1}}')
    print('')
    print(f'{"Achievement":<20}' + cols + r' \\')
    print('')
    wins = collections.defaultdict(int)
    for task in tasks:
        k = tasks.index(task)
        if task.startswith('achievement_'):
            name = task[len('achievement_'):].replace('_', ' ').title()
        else:
            name = task.replace('_', ' ').title()
        print(f'{name:<20}', end='')
        best = max(percents[methods.index(m), k] for m in legend.keys())
        for method in legend.keys():
            i = methods.index(method)
            value = percents[i][k]
            winner = value >= 0.95 * best and value > 0
            fmt = rf'{value:.1f}\%'
            fmt = (r'\o' if len(fmt) < 6 else ' ') + fmt
            fmt = rf'\b{{{fmt}}}' if winner else f'   {fmt} '
            if winner:
                wins[method] += 1
            print(rf' & ${fmt}$', end='')
        print(r' \\')
    print('')

    print(f'{"Score":<20}', end='')
    best = max(scores[methods.index(m)] for m in legend.keys())
    for method in legend.keys():
        value = scores[methods.index(method)]
        bold = value >= 0.95 * best and value > 0
        fmt = rf'{value:.1f}\%'
        fmt = (r'\o' if len(fmt) < 6 else ' ') + fmt
        fmt = rf'\b{{{fmt}}}' if bold else f'   {fmt} '
        print(rf' & ${fmt}$', end='')
    print(r' \\')


if __name__ == '__main__':
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
    print("Agent Scores Table:")
    print_scores(agent_dirs=in_dirs, legend=agent_legend, clip_seeds_to=clip_to_num_seeds)

    print("\nAbility Spectrum Table:")
    print_spectrum(agent_dirs=in_dirs, legend=agent_legend, clip_seeds_to=clip_to_num_seeds)
