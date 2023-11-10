import torch
from src.agents.dqn import DqnAgent
from src.utils.constant_builder import PathBuilder


class DoubleDqnAgent(DqnAgent):

    def __init__(
            self, epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma=0.92,
            epsilon_min=0.01, epsilon_dec=5e-7, replace=1000, hidden_units_conv: int = 16, algo: str = "doubleDqnAgent",
            env_name: str = "crafter", checkpoint_path=PathBuilder.DQN_AGENT_CHECKPOINT_DIR,
            number_of_frames_to_concatenate: int = 4
    ):
        super().__init__(
            epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma, epsilon_min,
            epsilon_dec, replace, hidden_units_conv, algo, env_name, checkpoint_path, number_of_frames_to_concatenate
        )

    def replace_target_network(self):
        if self.replace_target_count is not None and self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.take_memory()

        indices = torch.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]

        with torch.inference_mode():
            best_actions = self.q_eval.forward(states_).argmax(dim=1)
            q_next = self.q_next.forward(states_)
            q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, best_actions] * (1 - dones.float())

        loss = self.q_eval.loss(q_target, q_pred).to(self.device)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrease_epsilon()
