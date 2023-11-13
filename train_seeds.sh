#!/bin/sh

random="random_agent"
dqn="dqn_agent"
ddqn="doubledqn_agent"
dddqn="dueling_doubledqn_agent"

for i in $(seq 0 4); do
  python train.py --steps 100_000 --eval-interval 10_000 --logdir logdir/$dqn/"$i"
done
