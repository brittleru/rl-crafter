# Crafter Assignment starter code

This folder contains the following code:

- `train.py` A basic training loop with a random agent you can use for your own agent. Feel free to modify it at will.
- `src/crafter_wrapper.py` A wrapper over the `Crafter` environment that provides basic logging and observation preprocessing.
- `analysis/plot_eval_performance.py` A simple script for plotting the performance of your agent during evaluation (not training).

## Instructions

Follow the installation instructions in the [Crafter repository](https://github.com/danijar/crafter). It's ideal to use some kind of virtual env, my personal favourite is `miniconda`, although installation should work with the system's python as well.

For running the Random Agent execute:

```bash
python train.py --steps 10_000 --eval-interval 2500 --logdir logdir/random_agent/0
```

This will run the Random Agent for 10_000 steps and evaluate it every 2500 steps for 20 episodes. The results with be written in `logdir/random_agent/0`, where `0` indicates the run.

For executing multiple runs in parallel you could do:

```bash
for i in $(seq 1 4); do python train.py --steps 250_000 --eval-interval 25_000 --logdir logdir/random_agent/$i & done
```

### Visualization

Finally, you can visualize the _evaluation_ performance of the agent across the four runs using:

```bash
python analysis/plot_eval_performance.py --logdir logdir/random_agent
```

For other performance metrics see the [plotting scripts](https://github.com/danijar/crafter/tree/main/analysis) in the original Crafter repo.


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














