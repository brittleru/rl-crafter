import torch
import numpy as np

from src.utils.constant_builder import PathBuilder
from src.agents.dueling_dqn import DuelingDqnAgent


class DuelingDoubleDqnAgent(DuelingDqnAgent):
    def __init__(
            self, epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma=0.92,
            epsilon_min=0.01, epsilon_dec=5e-7, replace=1000, algo: str = "duelingDoubleDqnAgent",
            env_name: str = "crafter", checkpoint_path=PathBuilder.DUELING_DOUBLE_DQN_AGENT_CHECKPOINT_DIR,
            number_of_frames_to_concatenate: int = 4
    ):
        super().__init__(epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma,
                         epsilon_min, epsilon_dec, replace, algo, env_name, checkpoint_path,
                         number_of_frames_to_concatenate)

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        self.q_eval.optimizer.zero_grad()
        self.replace_target_network()

        states, actions, rewards, states_, dones = self.take_memory()
        indices = torch.arange(self.batch_size)
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = torch.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = torch.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]
        q_eval = torch.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        best_actions = torch.argmax(q_eval, dim=1)
        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[best_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrease_epsilon()
