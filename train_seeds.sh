#!/bin/sh

for i in $(seq 1 4)
do
  python train.py --steps 1_000 --eval-interval 250 --logdir logdir/random_agent/$i
done
