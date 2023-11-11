import torch

from src.agents.replay_buffer import ReplayBuffer
from src.networks.dqn import DeepQNetwork
from src.utils.constant_builder import PathBuilder


class DqnAgent(object):
    def __init__(
            self, epsilon, learning_rate, number_actions, input_sizes, memory_size, batch_size, device, gamma=0.92,
            epsilon_min=0.01, epsilon_dec=5e-7, replace=1000, hidden_units_conv: int = 16, algo: str = "dqnAgent",
            env_name: str = "crafter", checkpoint_path=PathBuilder.DQN_AGENT_CHECKPOINT_DIR,
            number_of_frames_to_concatenate: int = 4
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.number_actions = number_actions
        self.input_sizes = input_sizes
        self.batch_size = batch_size
        self.device = device
        self.epsilon_min = epsilon_min
        self.epsilon_dec = epsilon_dec
        self.replace_target_count = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_path = checkpoint_path
        self.action_space = [i for i in range(number_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(memory_size, input_sizes, number_of_frames_to_concatenate, self.device)
        self.q_eval = DeepQNetwork(
            number_actions=self.number_actions, input_size=self.input_sizes, learning_rate=self.learning_rate,
            checkpoint_name=f"{self.env_name}_{self.algo}_q_eval", checkpoint_path=self.checkpoint_path,
            device=self.device, epsilon_adam=1e-4, hidden_units_conv=hidden_units_conv
        )
        self.q_eval.view_model()
        self.q_next = DeepQNetwork(
            number_actions=self.number_actions, input_size=self.input_sizes, learning_rate=self.learning_rate,
            checkpoint_name=f"{self.env_name}_{self.algo}_q_next", checkpoint_path=self.checkpoint_path,
            device=self.device, epsilon_adam=1e-4, hidden_units_conv=hidden_units_conv
        )

    @torch.inference_mode()
    def act(self, observation):
        if torch.rand(1).item() > self.epsilon:
            state = torch.stack((observation,))
            actions = self.q_eval.forward(state=state)
            action = torch.argmax(actions).item()
        else:
            action = torch.randint(self.number_actions, (1,)).item()

        return action

    # TODO: add replay buffer to CPU not GPU (so you can parallelize)
    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state=state, action=action, reward=reward, state_=state_, done=done)

    def take_memory(self):
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)
        states = state
        actions = action
        rewards = reward
        states_ = state_
        dones = done

        return states, actions, rewards, states_, dones

    def decrease_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_dec
        else:
            self.epsilon = self.epsilon_min

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_count == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def set_train(self) -> None:
        self.q_eval.train()

    def set_eval(self) -> None:
        self.q_eval.eval()

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
