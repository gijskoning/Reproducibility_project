# Reproducibility project
This repository is an assignment of the Deep Learning course given by the TUDelft.
It reproduces the paper "Influence-aware Memory Architectures for Deep Reinforcement Learning" written by Miguel Suau ([link](https://arxiv.org/abs/1911.07643)).
The code for the original paper was developed in Tensorflow ([link](https://github.com/INFLUENCEorg/influence-aware-memory)).  This project recreates it in Pytorch and has some additional experiments and evaluations.   
The model will be evaluated using the PPO algorithm retrieved from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

Currently, the code is a work in progress and is only a not working copy of the tensorflow one.

## Executing the code
Currently the model can be tested using the environments:  
PongNoFrameskip-v4 (for quicker testing the num-processes is decreased from 8 to 1)
```
python main.py --env-name "PongNoFrameskip-v4" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 5 --use-linear-lr-decay --entropy-coef 0.01
```
The Warehouse environment included in the code of the paper.
```
python main.py --env-name "Warehouse" --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 0.5 --num-processes 1 --num-steps 128 --num-mini-batch 4 --log-interval 5 --use-linear-lr-decay --entropy-coef 0.01
```