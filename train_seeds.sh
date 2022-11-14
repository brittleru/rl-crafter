#!/bin/sh

random="random_agent"
dqn="dqn_agent"

for i in $(seq 1 4); do
  python train.py --steps 250_000 --eval-interval 25_000 --logdir logdir/$dqn/"$i"
done
