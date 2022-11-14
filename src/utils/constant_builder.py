import os
from pathlib import Path
from typing import Final

from src.utils.const import Const


class AgentTypes(Const):
    RANDOM: str = "random"
    DQN: str = "DQN"
    DDQN: str = "DDQN"
    RAINBOW: str = "RAINBOW"


class PathBuilder(Const):
    """
    This class creates all the path variables that are needed for this project. \n
    The paths works on both windows and unix operating systems. \n
    Available paths are for plots, logs, checkpoints and so on.
    """

    PROJECT_DIR: Final[str] = Path(__file__).resolve().parent.parent.parent

    ANALYSIS_DIR: Final[str] = os.path.join(PROJECT_DIR, "analysis")
    EVAL_PLOTS_DIR: Final[str] = os.path.join(ANALYSIS_DIR, "eval_plots")

    CHECKPOINTS_DIR: Final[str] = os.path.join(PROJECT_DIR, "checkpoints")
    DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(CHECKPOINTS_DIR, "dqn_agent")

    LOGDIR_DIR: Final[str] = os.path.join(PROJECT_DIR, "logdir")
    RANDOM_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "random_agent")
    DQN_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "dqn_agent")


if __name__ == "__main__":
    print(PathBuilder.PROJECT_DIR)
    print(PathBuilder.ANALYSIS_DIR)
    print(PathBuilder.EVAL_PLOTS_DIR)
    print(PathBuilder.CHECKPOINTS_DIR)
    print(PathBuilder.DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.LOGDIR_DIR)
    print(PathBuilder.RANDOM_AGENT_LOG_DIR)
    print(PathBuilder.DQN_AGENT_LOG_DIR)