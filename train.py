import os.path

import torch
import pickle
import argparse

from pathlib import Path

from src.agents.dqn import DqnAgent
from src.crafter_wrapper import Env
from src.agents.random import RandomAgent
from src.utils.constant_builder import AgentTypes
from src.utils.constant_builder import PathBuilder


def _save_stats(episodic_returns, crt_step, path):
    # save the evaluation stats
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    print(
        "[{:06d}] eval results: R/ep={:03.2f}, std={:03.2f}.".format(
            crt_step, avg_return, episodic_returns.std().item()
        )
    )
    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return}, f)


def eval(agent, env, crt_step, opt):
    """Use the greedy, deterministic policy, not the epsilon-greedy policy you might use during training."""
    episodic_returns = []
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward

    _save_stats(episodic_returns, crt_step, opt.logdir)


def _info(opt):
    try:
        int(opt.logdir.split("/")[-1])
    except:
        print(
            "Warning, logdir path should end in a number indicating a separate " +
            "training run, else the results might be overwritten."
        )
    if Path(opt.logdir).exists():
        print("Warning! Logdir path exists, results can be corrupted.")
    print(f"Saving results in {opt.logdir}.")
    print(f"Observations are of dims ({opt.history_length},64,64), with values between 0 and 1.")


def build_agent(environment: Env, device, agent_type: str = AgentTypes.DQN,
                checkpoint_path: str = os.path.join(PathBuilder.DQN_AGENT_CHECKPOINT_DIR, "0")):
    match agent_type:
        case AgentTypes.RANDOM:
            return RandomAgent(environment.action_space.n)
        case AgentTypes.DQN:
            return DqnAgent(
                epsilon=1,
                learning_rate=0.0000625,
                number_actions=environment.action_space.n,
                input_sizes=(environment.obs_dim, environment.obs_dim),  # check this if it's needed to be a tuple
                memory_size=100_000,
                batch_size=32,
                device=device,
                gamma=0.92,
                epsilon_min=0.1,
                epsilon_dec=1e-5,
                replace=1000,
                checkpoint_path=checkpoint_path
            )
        case AgentTypes.DDQN:
            raise NotImplementedError(f"{AgentTypes.DDQN} not implemented yet")
        case AgentTypes.RAINBOW:
            raise NotImplementedError(f"{AgentTypes.RAINBOW} not implemented yet")


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = build_agent(environment=env, device=opt.device, agent_type=opt.agent_type)

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.act(obs)
        obs, reward, done, info = env.step(action)

        step_cnt += 1

        if step_cnt % opt.eval_interval == 0:
            print(f"[{step_cnt: 06d}] progress={(100.0 * step_cnt / opt.steps): 03.2f}%.")

        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            eval(agent, eval_env, step_cnt, opt)


def get_options():
    """
        Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For devel purposes feel free to change the number of training steps and
        the evaluation interval.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="logdir/dqn_agent/0")
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=1_000_000,
        help="Total number of training steps.",
    )
    parser.add_argument(
        "-hist-len",
        "--history-length",
        default=4,
        type=int,
        help="The number of frames to stack when creating an observation.",
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100_000,
        metavar="STEPS",
        help="Number of training steps between evaluations",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=20,
        metavar="N",
        help="Number of evaluation episodes to average over",
    )
    parser.add_argument(
        "--agent-type",
        type=str,
        default=AgentTypes.DQN,
        help="Type of agent architecture"
    )

    return parser.parse_args()


if __name__ == "__main__":
    main(get_options())
