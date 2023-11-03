import torch


class RandomAgent:
    """
    An example Random Agent
    """

    def __init__(self, action_num) -> None:
        self.action_num = action_num
        # a uniformly random policy
        self.policy = torch.distributions.Categorical(torch.ones(action_num) / action_num)

    def act(self, observation: torch.Tensor):
        """Since this is a random agent the observation is not used."""
        return self.policy.sample().item()
