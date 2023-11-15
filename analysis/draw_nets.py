import os.path

import torch
from torchviz import make_dot

from src.networks.dqn import DeepQNetwork
from src.utils.constant_builder import PathBuilder
from train import load_hyperparameters


def draw_torchviz():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper_parameters = load_hyperparameters(PathBuilder.HYPER_PARAMETERS_PATH)
    learning_rate = hyper_parameters['learning_rate']
    hidden_units_conv = hyper_parameters['hidden_units_conv']
    input_shape = (64, 64)
    checkpoint_temp_path = os.path.join(PathBuilder.DQN_AGENT_CHECKPOINT_DIR, "99")
    dqn = DeepQNetwork(
        number_actions=17, input_size=input_shape, learning_rate=learning_rate,
        checkpoint_name=f"dqn_model", checkpoint_path=checkpoint_temp_path,
        device=device, epsilon_adam=1e-4, hidden_units_conv=hidden_units_conv
    )
    input_image = torch.rand((1, 4, *input_shape), dtype=torch.float64).to(device)
    y = dqn.forward(input_image)
    make_dot(y, params=dict(dqn.named_parameters())).render("DeepQNetwork-Architecture", format="png")


def create_onnx_for_netron():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hyper_parameters = load_hyperparameters(PathBuilder.HYPER_PARAMETERS_PATH)
    learning_rate = hyper_parameters['learning_rate']
    hidden_units_conv = hyper_parameters['hidden_units_conv']
    input_shape = (64, 64)
    checkpoint_temp_path = os.path.join(PathBuilder.DQN_AGENT_CHECKPOINT_DIR, "99")
    dqn = DeepQNetwork(
        number_actions=17, input_size=input_shape, learning_rate=learning_rate,
        checkpoint_name=f"dqn_model", checkpoint_path=checkpoint_temp_path,
        device=device, epsilon_adam=1e-4, hidden_units_conv=hidden_units_conv
    )
    input_image = torch.rand((1, 4, *input_shape), dtype=torch.float64).to(device)

    torch.onnx.export(dqn, input_image, "model.onnx", verbose=True)


if __name__ == '__main__':
    draw_torchviz()
    create_onnx_for_netron()