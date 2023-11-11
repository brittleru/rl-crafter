# Crafter - Reinforcement Learning 
This repository contains results and code for training a few deep neural networks for reinforcement learning for the 
Crafter game made by Danijar Hafner, more info on [Crafter Repository](https://github.com/danijar/crafter). If you
wish to experiment yourself you can modify the [hyperparameters.json](hyperparameters.json). More info for training, 
visualisation and the environment can be found on [the initial assignment readme file](assignment_initial.md) 

## Setup
It is required to have **Python v3.10+** and install the libraries from [requirements.txt](requirements.txt).

## Approach
The task is to train a **deep reinforcement learning** algorithm that performs better than a **random agent** which 
takes random decisions at each move. The random agent performs at **worst 0.7** and at **best 1.8** average reward per 
episode. When playing the game we've seen that is difficult even for a human to beat this game, the difficulties were
surviving during the night and eating to avoid starvation (gardening was necessary for this since cows were limited). 

We chose to start small with a DQN (Deep Q Network) and optimize the hyperparameters until the agent performed better
than the random agent. Since each input to the model are 4 gray scaled images, a CNN architecture was chosen for the
backbone of the DQN agent, the CNN architecture is similar with TinyVGG, then at the classifier layer we used two
fully connected layers. 

(TODO: plot some visualization of the network)

After having a strong baseline agent, we trained better versions of that agent with the same hyperparameters and 
backbone neural network architectures such as:
* DDQN (Double DQN)
* Dueling DQN
* Dueling DDQN 

## Results
To visualize some results we chose to plot the average episodic reward for two or three seeds, the success rate for 
each achievement of each agent as a table and as a histogram. (TODO: see if add some achievement curves) 

(TODO: plot some visualization of the results)


## Hacking the environment with heuristics
Since in a real-world RL scenario heuristics such as, avoid doing the same task over and over or find the resources 
needed to avoid dying, would be used, we decided to also implement some.
* TODO: add here some heuristics used














