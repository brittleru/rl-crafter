import os
from pathlib import Path
from typing import Final

from src.utils.const import Const


class PathBuilder(Const):
    """
    This class creates all the path variables that are needed for this project. \n
    The paths works on both windows and unix operating systems. \n
    Available paths are for plots, logs, checkpoints and so on.
    """

    PROJECT_DIR: Final[str] = Path(__file__).resolve().parent.parent.parent

    ANALYSIS_DIR: Final[str] = os.path.join(PROJECT_DIR, "analysis")
    EVAL_PLOTS_DIR: Final[str] = os.path.join(ANALYSIS_DIR, "eval_plots")
    TRAIN_PLOTS_DIR: Final[str] = os.path.join(ANALYSIS_DIR, "train_plots")
    SCORE_PLOTS_DIR: Final[str] = os.path.join(ANALYSIS_DIR, "score_plots")
    ACHIEVEMENT_PLOTS_DIR: Final[str] = os.path.join(ANALYSIS_DIR, "achievement_plots")

    CHECKPOINTS_DIR: Final[str] = os.path.join(PROJECT_DIR, "checkpoints")
    DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(CHECKPOINTS_DIR, "dqn_agent")
    DOUBLE_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(CHECKPOINTS_DIR, "doubledqn_agent")
    DUELING_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(CHECKPOINTS_DIR, "duelingdqn_agent")
    DUELING_DOUBLE_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(CHECKPOINTS_DIR, "dueling_doubledqn_agent")
    RAINBOW_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(CHECKPOINTS_DIR, "rainbow_agent")

    LOGDIR_DIR: Final[str] = os.path.join(PROJECT_DIR, "logdir")
    RANDOM_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "random_agent")
    DQN_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "dqn_agent")
    DOUBLE_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "doubledqn_agent")
    DUELING_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "duelingdqn_agent")
    DUELING_DOUBLE_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "dueling_doubledqn_agent")
    RAINBOW_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(LOGDIR_DIR, "rainbow_agent")

    HACKED_CHECKPOINTS_DIR: Final[str] = os.path.join(CHECKPOINTS_DIR, "hacked")
    HACKED_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(HACKED_CHECKPOINTS_DIR, "dqn_agent")
    HACKED_DOUBLE_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(HACKED_CHECKPOINTS_DIR, "doubledqn_agent")
    HACKED_DUELING_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(HACKED_CHECKPOINTS_DIR, "duelingdqn_agent")
    HACKED_DUELING_DOUBLE_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(HACKED_CHECKPOINTS_DIR, "dueling_doubledqn_agent")
    HACKED_RAINBOW_DQN_AGENT_CHECKPOINT_DIR: Final[str] = os.path.join(HACKED_CHECKPOINTS_DIR, "rainbow_agent")

    HACKED_LOGDIR_DIR: Final[str] = os.path.join(LOGDIR_DIR, "hacked")
    HACKED_RANDOM_AGENT_LOG_DIR: Final[str] = os.path.join(HACKED_LOGDIR_DIR, "random_agent")
    HACKED_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(HACKED_LOGDIR_DIR, "dqn_agent")
    HACKED_DOUBLE_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(HACKED_LOGDIR_DIR, "doubledqn_agent")
    HACKED_DUELING_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(HACKED_LOGDIR_DIR, "duelingdqn_agent")
    HACKED_DUELING_DOUBLE_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(HACKED_LOGDIR_DIR, "dueling_doubledqn_agent")
    HACKED_RAINBOW_DQN_AGENT_LOG_DIR: Final[str] = os.path.join(HACKED_LOGDIR_DIR, "rainbow_agent")

    HYPER_PARAMETERS_PATH: Final[str] = os.path.join(PROJECT_DIR, "hyperparameters.json")


class AgentTypes(Const):
    RANDOM: str = "random"
    DQN: str = "DQN"
    DDQN: str = "DDQN"
    RAINBOW: str = "RAINBOW"
    DUELING_DQN: str = "DUELING-DQN"
    DUELING_DOUBLE_DQN: str = "DUELING-DOUBLE-DQN"

    PATH_TO_NAME: dict = {
        PathBuilder.RANDOM_AGENT_LOG_DIR: RANDOM,
        PathBuilder.DQN_AGENT_LOG_DIR: DQN,
        PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR: DDQN,
        PathBuilder.DUELING_DQN_AGENT_LOG_DIR: DUELING_DQN,
        PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR: DUELING_DOUBLE_DQN,
        PathBuilder.RAINBOW_DQN_AGENT_LOG_DIR: RAINBOW
    }

    HACKED_DQN: str = "HACKED-DQN"


if __name__ == "__main__":
    print(PathBuilder.PROJECT_DIR)
    print(PathBuilder.ANALYSIS_DIR)
    print(PathBuilder.EVAL_PLOTS_DIR)
    print(PathBuilder.CHECKPOINTS_DIR)
    print(PathBuilder.DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.DOUBLE_DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.LOGDIR_DIR)
    print(PathBuilder.RANDOM_AGENT_LOG_DIR)
    print(PathBuilder.DQN_AGENT_LOG_DIR)
    print(PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR)
    print(PathBuilder.DUELING_DQN_AGENT_LOG_DIR)
    print(PathBuilder.DUELING_DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR)
    print(PathBuilder.DUELING_DOUBLE_DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.RAINBOW_DQN_AGENT_LOG_DIR)
    print(PathBuilder.RAINBOW_DQN_AGENT_CHECKPOINT_DIR)

    print(PathBuilder.HACKED_CHECKPOINTS_DIR)
    print(PathBuilder.HACKED_DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.HACKED_DOUBLE_DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.HACKED_LOGDIR_DIR)
    print(PathBuilder.HACKED_RANDOM_AGENT_LOG_DIR)
    print(PathBuilder.HACKED_DQN_AGENT_LOG_DIR)
    print(PathBuilder.HACKED_DOUBLE_DQN_AGENT_LOG_DIR)
    print(PathBuilder.HACKED_DUELING_DQN_AGENT_LOG_DIR)
    print(PathBuilder.HACKED_DUELING_DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.HACKED_DUELING_DOUBLE_DQN_AGENT_LOG_DIR)
    print(PathBuilder.HACKED_DUELING_DOUBLE_DQN_AGENT_CHECKPOINT_DIR)
    print(PathBuilder.HACKED_RAINBOW_DQN_AGENT_LOG_DIR)
    print(PathBuilder.HACKED_RAINBOW_DQN_AGENT_CHECKPOINT_DIR)

    print(AgentTypes.PATH_TO_NAME)
