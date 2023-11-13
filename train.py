import argparse
import json
import os.path
import pickle
from pathlib import Path
from time import time

import torch

from src.agents.ddqn import DoubleDqnAgent
from src.agents.dqn import DqnAgent
from src.agents.dueling_ddqn import DuelingDoubleDqnAgent
from src.agents.dueling_dqn import DuelingDqnAgent
from src.agents.rainbow_dqn import RainbowDqnAgent
from src.agents.random import RandomAgent
from src.crafter_wrapper import Env
from src.utils.constant_builder import AgentTypes
from src.utils.constant_builder import PathBuilder
from src.utils.plot_utils import plot_rewards, plot_epsilon
from src.utils.train_utils import display_readable_time, get_readable_time


def load_hyperparameters(file_path):
    with open(file_path, 'r') as file:
        hyperparameters = json.load(file)
    return hyperparameters


def _save_stats(episodic_returns, crt_step, path, eval_time: str | None = None):
    """ Save the evaluation stats """
    episodic_returns = torch.tensor(episodic_returns)
    avg_return = episodic_returns.mean().item()
    min_return = episodic_returns.min().item()
    max_return = episodic_returns.max().item()

    stats_info = f"[{crt_step:06d}] eval results: R/ep={avg_return:03.2f}, std={episodic_returns.std().item():03.2f}."

    if eval_time is not None:
        stats_info += f" eval time: {eval_time}"

    print(stats_info)

    with open(path + "/eval_stats.pkl", "ab") as f:
        pickle.dump({"step": crt_step, "avg_return": avg_return, "min_return": min_return, "max_return": max_return}, f)


def eval(agent, env, crt_step, opt):
    """Use the greedy, deterministic policy, not the epsilon-greedy policy you might use during training."""
    episodic_returns = []
    eval_start_time = time()
    for _ in range(opt.eval_episodes):
        obs, done = env.reset(), False
        episodic_returns.append(0)
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            episodic_returns[-1] += reward
    eval_time = get_readable_time(eval_start_time, time())

    _save_stats(episodic_returns, crt_step, opt.logdir, eval_time)


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
    print(f"Observations are of dims ({opt.history_length}, 64, 64), with values between 0 and 1.")


def build_agent(
        environment: Env, device, agent_type: str = AgentTypes.DQN,
        checkpoint_path: str = os.path.join(PathBuilder.DQN_AGENT_CHECKPOINT_DIR, "99")
):
    hyper_parameters = load_hyperparameters(PathBuilder.HYPER_PARAMETERS_PATH)
    epsilon = hyper_parameters['epsilon']
    learning_rate = hyper_parameters['learning_rate']
    memory_size = hyper_parameters['memory_size']
    batch_size = hyper_parameters['batch_size']
    gamma = hyper_parameters['gamma']
    epsilon_min = hyper_parameters['epsilon_min']
    epsilon_dec = hyper_parameters['epsilon_dec']
    replace = hyper_parameters['replace']
    hidden_units_conv = hyper_parameters['hidden_units_conv']

    match agent_type:
        case AgentTypes.RANDOM:
            print(f"Using {AgentTypes.RANDOM} architecture")
            return RandomAgent(environment.action_space.n)
        case AgentTypes.DQN:
            print(f"Using {AgentTypes.DQN} architecture")
            return DqnAgent(
                epsilon=epsilon,
                learning_rate=learning_rate,
                number_actions=environment.action_space.n,
                input_sizes=(environment.obs_dim, environment.obs_dim),
                memory_size=memory_size,
                batch_size=batch_size,
                device=device,
                gamma=gamma,
                epsilon_min=epsilon_min,
                epsilon_dec=epsilon_dec,
                replace=replace,
                hidden_units_conv=hidden_units_conv,
                checkpoint_path=checkpoint_path
            )
        case AgentTypes.DDQN:
            print(f"Using {AgentTypes.DDQN} architecture")
            return DoubleDqnAgent(
                epsilon=epsilon,
                learning_rate=learning_rate,
                number_actions=environment.action_space.n,
                input_sizes=(environment.obs_dim, environment.obs_dim),
                memory_size=memory_size,
                batch_size=batch_size,
                device=device,
                gamma=gamma,
                epsilon_min=epsilon_min,
                epsilon_dec=epsilon_dec,
                replace=replace,
                hidden_units_conv=hidden_units_conv,
                checkpoint_path=checkpoint_path
            )
        case AgentTypes.DUELING_DQN:
            print(f"Using {AgentTypes.DUELING_DQN} architecture")
            return DuelingDqnAgent(
                epsilon=epsilon,
                learning_rate=learning_rate,
                number_actions=environment.action_space.n,
                input_sizes=(environment.obs_dim, environment.obs_dim),
                memory_size=memory_size,
                batch_size=batch_size,
                device=device,
                gamma=gamma,
                epsilon_min=epsilon_min,
                epsilon_dec=epsilon_dec,
                replace=replace,
                hidden_units_conv=hidden_units_conv,
                checkpoint_path=checkpoint_path
            )
        case AgentTypes.DUELING_DOUBLE_DQN:
            print(f"Using {AgentTypes.DUELING_DOUBLE_DQN} architecture")
            return DuelingDoubleDqnAgent(
                epsilon=epsilon,
                learning_rate=learning_rate,
                number_actions=environment.action_space.n,
                input_sizes=(environment.obs_dim, environment.obs_dim),
                memory_size=memory_size,
                batch_size=batch_size,
                device=device,
                gamma=gamma,
                epsilon_min=epsilon_min,
                epsilon_dec=epsilon_dec,
                replace=replace,
                hidden_units_conv=hidden_units_conv,
                checkpoint_path=checkpoint_path
            )
        case AgentTypes.RAINBOW:
            print(f"Using {AgentTypes.RAINBOW} architecture")
            return RainbowDqnAgent(
                epsilon=epsilon,
                learning_rate=learning_rate,
                number_actions=environment.action_space.n,
                input_sizes=(environment.obs_dim, environment.obs_dim),
                memory_size=memory_size,
                batch_size=batch_size,
                device=device,
                gamma=gamma,
                epsilon_min=epsilon_min,
                epsilon_dec=epsilon_dec,
                replace=replace,
                hidden_units_conv=hidden_units_conv,
                checkpoint_path=checkpoint_path
            )


def main(opt):
    _info(opt)
    opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {str(opt.device).upper()} device...")

    env = Env("train", opt)
    eval_env = Env("eval", opt)
    agent = build_agent(environment=env, device=opt.device, agent_type=opt.agent_type, checkpoint_path=opt.check_dir)
    model_number = os.path.basename(opt.check_dir)

    if opt.agent_type is not AgentTypes.RANDOM:
        agent.show_architecture()

    # main loop
    ep_cnt, step_cnt, done = 0, 0, True
    score_hist, epsilon_hist, step_hist = [], [], []
    start_time = time()
    score = 0
    print("\nStarting training...")
    start_percent_eval_total_time = time()

    while step_cnt < opt.steps or not done:
        if done:
            ep_cnt += 1
            obs, done = env.reset(), False

        action = agent.act(obs)
        obs_, reward, done, info = env.step(action)

        if opt.agent_type is not AgentTypes.RANDOM:
            agent.store_transition(state=obs, action=action, reward=reward, state_=obs_, done=done)
            agent.learn()

        obs = obs_
        score += reward
        step_cnt += 1

        if step_cnt % opt.eval_interval == 0:
            end_percent_eval_total_time = time()
            percent_eval_total_time = get_readable_time(start_percent_eval_total_time, end_percent_eval_total_time)
            print(f"[{step_cnt:06d}] At {(100.0 * step_cnt / opt.steps):03.2f}%. "
                  f"Time for {opt.eval_interval} steps: {percent_eval_total_time}")

        if opt.agent_type is not AgentTypes.RANDOM:
            epsilon_hist.append(agent.epsilon)
        step_hist.append(step_cnt)
        score_hist.append(score)
        # evaluate once in a while
        if step_cnt % opt.eval_interval == 0:
            if opt.agent_type is not AgentTypes.RANDOM:
                agent.set_eval()
            eval(agent, eval_env, step_cnt, opt)
            if opt.agent_type is not AgentTypes.RANDOM:
                agent.set_train()
            start_percent_eval_total_time = time()

    display_readable_time(start_time, time())
    plot_rewards(steps=step_hist, scores=score_hist, network_type=opt.agent_type, model_num=model_number)
    if opt.agent_type is not AgentTypes.RANDOM:
        plot_epsilon(steps=step_hist, epsilons=epsilon_hist, network_type=opt.agent_type, model_num=model_number)
    agent.save_models()


def get_options(
        agent_type: str = AgentTypes.DQN, log_dir: str = os.path.join(PathBuilder.DQN_AGENT_LOG_DIR, "99"),
        checkpoint_dir: str = os.path.join(PathBuilder.DQN_AGENT_CHECKPOINT_DIR, "99")
):
    """
        Configures a parser. Extend this with all the best performing hyperparameters of
        your agent as defaults.

        For development purposes feel free to change the number of training steps and
        the evaluation interval.

        The default agent is DQN.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default=log_dir)
    parser.add_argument(
        "--steps",
        type=int,
        metavar="STEPS",
        default=100_000,
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
        default=5_000,
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
        default=agent_type,
        help="Type of agent architecture"
    )
    parser.add_argument(
        "--check-dir",
        type=str,
        default=checkpoint_dir,
        help="Directory to safe the model checkpoints"
    )

    return parser.parse_args()


if __name__ == "__main__":
    # RANDOM
    # AgentTypes.RANDOM
    log_random_path = os.path.join(PathBuilder.RANDOM_AGENT_LOG_DIR, "99")

    # DQN
    # AgentTypes.DQN
    log_dqn_path = os.path.join(PathBuilder.DQN_AGENT_LOG_DIR, "99")
    checkpoint_dqn_path = os.path.join(PathBuilder.DQN_AGENT_CHECKPOINT_DIR, "99")

    # Double DQN
    # AgentTypes.DDQN
    log_ddqn_path = os.path.join(PathBuilder.DOUBLE_DQN_AGENT_LOG_DIR, "99")
    checkpoint_ddqn_path = os.path.join(PathBuilder.DOUBLE_DQN_AGENT_CHECKPOINT_DIR, "99")

    # Dueling DQN
    # AgentTypes.DUELING_DQN
    log_duel_dqn_path = os.path.join(PathBuilder.DUELING_DQN_AGENT_LOG_DIR, "99")
    checkpoint_duel_dqn_path = os.path.join(PathBuilder.DUELING_DQN_AGENT_CHECKPOINT_DIR, "99")

    # Dueling Double DQN
    # AgentTypes.DUELING_DOUBLE_DQN
    log_duel_double_dqn_path = os.path.join(PathBuilder.DUELING_DOUBLE_DQN_AGENT_LOG_DIR, "99")
    checkpoint_duel_double_dqn_path = os.path.join(PathBuilder.DUELING_DOUBLE_DQN_AGENT_CHECKPOINT_DIR, "99")

    # Rainbow DQN
    # AgentTypes.RAINBOW
    log_rainbow_dqn_path = os.path.join(PathBuilder.RAINBOW_DQN_AGENT_LOG_DIR, "2")
    checkpoint_rainbow_dqn_path = os.path.join(PathBuilder.RAINBOW_DQN_AGENT_CHECKPOINT_DIR, "2")

    main(get_options(
        agent_type=AgentTypes.RAINBOW,
        log_dir=log_rainbow_dqn_path,
        checkpoint_dir=checkpoint_rainbow_dqn_path
    ))
