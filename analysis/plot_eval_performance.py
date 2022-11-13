import os
import pickle
import pathlib
import argparse

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path


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
    # TODO: change figure plot name
    plt.savefig(os.path.join(PLOT_EVAL_PATH, "demo_plot.png"))
    plt.show()


if __name__ == "__main__":
    PROJECT_PATH = Path(__file__).resolve().parent.parent
    ANALYSIS_PATH = os.path.join(PROJECT_PATH, "analysis")
    PLOT_EVAL_PATH = os.path.join(ANALYSIS_PATH, "eval_plots")
    LOGDIR_PATH = os.path.join(PROJECT_PATH, "logdir")
    RANDOM_AGENT_PATH = os.path.join(LOGDIR_PATH, "random_agent")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        default=RANDOM_AGENT_PATH,
        help="Path to the folder containing different runs.",
    )
    cfg = parser.parse_args()

    read_crafter_logs(cfg.logdir)
