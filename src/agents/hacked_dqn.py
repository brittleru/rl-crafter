from typing import List

import torch

from src.agents.dqn import DqnAgent
from src.utils.constant_builder import PathBuilder
from crafter import constants


class HackedDqnAgent(DqnAgent):
    def __init__(
            self, epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma=0.92,
            epsilon_min=0.01, epsilon_dec=5e-7, replace=1000, hidden_units_conv: int = 16, algo: str = "hackDqnAgent",
            env_name: str = "crafter", checkpoint_path=PathBuilder.DQN_AGENT_CHECKPOINT_DIR,
            number_of_frames_to_concatenate: int = 4
    ):
        super().__init__(
            epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma, epsilon_min,
            epsilon_dec, replace, hidden_units_conv, algo, env_name, checkpoint_path, number_of_frames_to_concatenate
        )

        self.action_to_idx = {action: num_action for action, num_action in zip(constants.actions, self.action_space)}

    @torch.inference_mode()
    def act(self, observation, info):
        """
        Name of actions are from the data.yaml from the crafter library. A corespondent mapping can be:
        {
            'noop': 0, 'move_left': 1, 'move_right': 2, 'move_up': 3, 'move_down': 4, 'do': 5, 'sleep': 6,
            'place_stone': 7, 'place_table': 8, 'place_furnace': 9, 'place_plant': 10, 'make_wood_pickaxe': 11,
            'make_stone_pickaxe': 12, 'make_iron_pickaxe': 13, 'make_wood_sword': 14, 'make_stone_sword': 15,
            'make_iron_sword': 16
        }
        """
        if torch.rand(1).item() > self.epsilon:
            state = torch.stack((observation,))
            actions = self.q_eval.forward(state=state)
            if info is not "":
                actions = self.check_actions(actions, info)
            action = torch.argmax(actions).item()
            return action

        action = torch.randint(self.number_actions, (1,)).item()
        return action

    def check_actions(self, actions, info: dict):
        pred_action = torch.argmax(actions).item()
        if self.action_to_idx[pred_action] == "make_wood_pickaxe":
            # TODO: check if achievement info['achievements'] done if done then skip this action and go for the next
            #  if not done check wood info['inventory'] and build

            ...
        # TODO: Similarly for the rest of the actions

        # TODO: If no action inferred after the checker pick the most likely action but not
        #  from the action pool from above.

        # TODO: If player is starving search for food
        # TODO: If player needs water and water below 7 drink more to avoid dying of dehydration.

        # TODO: check hacking the crafter wrapper.
        return actions

    def learn(self):
        if self.memory.memory_counter < self.batch_size:
            return

        self.replace_target_network()
        states, actions, rewards, states_, dones = self.take_memory()

        indices = torch.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]

        with torch.inference_mode():
            q_next = self.q_next.forward(states_).max(dim=1)[0]
            q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next * (1 - dones.float())

        loss = self.q_eval.loss(q_target, q_pred).to(self.device)
        self.q_eval.optimizer.zero_grad()
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
        self.decrease_epsilon()
