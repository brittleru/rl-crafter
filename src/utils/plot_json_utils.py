import json
import pathlib
from typing import List

from src.utils.constant_builder import AgentTypes


def read_jsonl(file_path: str | pathlib.Path):
    with open(file_path, "r") as jsonl_file:
        json_lines = list(jsonl_file)
        jsonl_file.close()

    data = []
    for json_line in json_lines:
        json_data = json.loads(json_line)
        data.append(json_data)
    return data


def compile_json_seeds(agent_dir: str, agent_steps: int | None = None):
    in_dir_path = pathlib.Path(agent_dir)
    filenames = sorted(list(in_dir_path.glob("**\\*\\stats.jsonl")))
    runs = []
    for seed, file_name in enumerate(filenames):
        seed_data = {"algorithm": AgentTypes.PATH_TO_NAME[agent_dir], "seed": seed}
        model_stats = read_jsonl(file_path=file_name)

        if agent_steps is not None:
            num_steps = len(model_stats)
            step_size = agent_steps / (num_steps - 1)
            xs_arr = [int(i * step_size) for i in range(num_steps)]
            seed_data["xs"] = xs_arr

        for iter_stats in model_stats:
            for key, value in iter_stats.items():
                if key in seed_data:
                    seed_data[key].append(value)
                else:
                    seed_data[key] = [value]
        runs.append(seed_data)

    return runs


def compile_all_agents_seeds(agent_dirs: List[str], clip_seeds_to: int = 3, agent_steps: int | None = None):
    total_runs = []

    for agent_dir in agent_dirs:
        agent_runs = compile_json_seeds(agent_dir=agent_dir, agent_steps=agent_steps)
        total_runs.extend(agent_runs[:clip_seeds_to])

    return total_runs
