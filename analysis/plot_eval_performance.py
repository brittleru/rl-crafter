import argparse
import os
import pathlib
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.utils.constant_builder import PathBuilder
from src.utils.string_utils import get_dir_name_as_img, get_dirs_name_as_img


def read_pkl(path):
    events = []
    with (open(path, "rb")) as openfile:
        while True:
            try:
                events.append(pickle.load(openfile))
            except EOFError:
                break
    return events


def read_crafter_logs(in_dir, clip=True):
    in_dir = pathlib.Path(in_dir)
    # read the pickles
    filenames = sorted(list(in_dir.glob("**\\*\\eval_stats.pkl")))
    runs = []
    for idx, fn in enumerate(filenames):
        df = pd.DataFrame(columns=["step", "avg_return"], data=read_pkl(fn))
        df["run"] = idx
        runs.append(df)

    # some runs might not have finished, and you might want to clip all of them to the shortest one.
    if clip:
        min_len = min([len(run) for run in runs])
        runs = [run[:min_len] for run in runs]
        print(f"Clipped all runs to {min_len}.")

    # plot
    df = pd.concat(runs, ignore_index=True)
    sns.lineplot(x="step", y="avg_return", data=df)
    plt.grid(visible=True)
    plt.savefig(os.path.join(PathBuilder.EVAL_PLOTS_DIR, get_dir_name_as_img(in_dir.__str__())))
    plt.show()


def compare_model_logs(in_dirs: list, clip=True):
    model_names = []
    agent_colors = {}

    for in_dir in in_dirs:
        in_dir = pathlib.Path(in_dir)
        model_name = os.path.basename(os.path.normpath(in_dir))
        model_names.append(model_name)
        # read the pickles
        filenames = sorted(list(in_dir.glob("**\\*\\eval_stats.pkl")))
        runs = []
        for idx, fn in enumerate(filenames):
            df = pd.DataFrame(columns=["step", "avg_return"], data=read_pkl(fn))
            df["run"] = idx
            runs.append(df)

        if clip:
            min_len = min([len(run) for run in runs])
            runs = [run[:min_len] for run in runs]
            print(f"Clipped all runs to {min_len}.")

        df = pd.concat(runs, ignore_index=True)
        line = sns.lineplot(x="step", y="avg_return", data=df)
        line_color = line.get_lines()[-1].get_color()
        agent_colors[model_name] = line_color

    handles = [plt.Line2D([0], [0], color=agent_colors[model_name], label=model_name) for model_name in model_names]
    plt.legend(handles=handles, title="Agent name")

    plt.grid(visible=True)
    plt.title("Average episodic reward for all agents")
    plt.savefig(os.path.join(PathBuilder.EVAL_PLOTS_DIR, get_dirs_name_as_img(in_dirs)))
    plt.show()


def get_options():
    """
    Log dirs enums:
        PathBuilder.RANDOM_AGENT_LOG_DIR
        PathBuilder.DQN_AGENT_LOG_DIR
        PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR
        PathBuilder.DUELING_DQN_AGENT_LOG_DIR
        PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR
        PathBuilder.RAINBOW_DQN_AGENT_LOG_DIR
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default=PathBuilder.DQN_AGENT_LOG_DIR,
        help="Path to the folder containing different runs.",
    )

    parser.add_argument(
        "--do-comparison",
        type=bool,
        default=False,
        help="Flag to choose if plot the all the trained models."
    )

    return parser.parse_args()


def run_eval(options):
    read_crafter_logs(options.logdir)

    if options.do_comparison:
        dirs = [
            PathBuilder.RANDOM_AGENT_LOG_DIR,
            PathBuilder.DQN_AGENT_LOG_DIR,
            PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR,
            PathBuilder.DUELING_DQN_AGENT_LOG_DIR,
            PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR
        ]
        compare_model_logs(dirs)


if __name__ == "__main__":
    run_eval(get_options())
