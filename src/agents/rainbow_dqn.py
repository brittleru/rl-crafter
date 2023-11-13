import torch
from src.networks.c51_dqn import C51DQN
from src.agents.dueling_dqn import DuelingDqnAgent
from src.utils.constant_builder import PathBuilder


class RainbowDqnAgent(DuelingDqnAgent):
    def __init__(
            self, epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma=0.92,
            epsilon_min=0.01, epsilon_dec=5e-7, replace=1000, hidden_units_conv: int = 16,
            algo: str = "rainbowDqnAgent", env_name: str = "crafter",
            checkpoint_path=PathBuilder.RAINBOW_DQN_AGENT_CHECKPOINT_DIR,
            number_of_frames_to_concatenate: int = 4
    ):
        super().__init__(
            epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma, epsilon_min,
            epsilon_dec, replace, hidden_units_conv, algo, env_name, checkpoint_path, number_of_frames_to_concatenate
        )

        self.q_eval = C51DQN(
            number_actions=self.number_actions, input_size=self.input_sizes, num_atoms=51,
            learning_rate=self.learning_rate, checkpoint_name=f"{self.env_name}_{self.algo}_q_eval",
            checkpoint_path=self.checkpoint_path, device=self.device
        )

        self.q_next = C51DQN(
            number_actions=self.number_actions, input_size=self.input_sizes, num_atoms=51,
            learning_rate=self.learning_rate, checkpoint_name=f"{self.env_name}_{self.algo}_q_next",
            checkpoint_path=self.checkpoint_path, device=self.device
        )

    @torch.inference_mode()
    def act(self, observation):
        if torch.rand(1).item() > self.epsilon:
            state = torch.stack((observation,))
            q_distribution = self.q_eval.forward(state=state)
            expected_values = torch.sum(q_distribution * self.q_eval.supports, dim=2)
            action = torch.argmax(expected_values[0]).item()
        else:
            action = torch.randint(self.number_actions, (1,)).item()

        return action

    def replace_target_network(self):
        if self.replace_target_count is not None and self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.take_memory()

        indices = torch.arange(self.batch_size)
        q_distribution = self.q_eval.forward(states)

        with torch.inference_mode():
            q_next_distribution = self.q_next.forward(states_)
            q_next = q_next_distribution.max(dim=1)[0]
            q_eval = q_distribution
            best_actions = torch.argmax(q_eval, dim=1)
            q_next[dones] = 0.0

        q_pred = q_distribution[indices, actions]
        q_next_values = torch.gather(q_next_distribution, 1, best_actions.unsqueeze(1))
        q_next_values = q_next_values.view(-1, self.q_eval.num_atoms)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)
        q_target = rewards + self.gamma * q_next_values * (1 - dones.float())

        loss = self.q_eval.c51_loss(q_target, q_pred).to(self.device)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrease_epsilon()
