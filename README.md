# Reproducibility project
This repository is an assignment of the Deep Learning course given by the TUDelft.
It reproduces the paper "Influence-aware Memory Architectures for Deep Reinforcement Learning" written by Miguel Suau ([link](https://arxiv.org/abs/1911.07643)).
The code for the original paper was developed in Tensorflow ([link](https://github.com/INFLUENCEorg/influence-aware-memory)).  This project recreates it in Pytorch and has some additional experiments and evaluations.   
The model will be evaluated using the PPO algorithm retrieved from: https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail

Currently, the code is a work in progress and is only a not working copy of the tensorflow one.

## Executing the code
The Warehouse environment included in the code of the paper.
*Todo the hyperparameter are not yet correct!*
Currently these parameters are correct: 
- learning rate, num epoch/num-steps, value coeff/value-loss-coef, entropy/entropy-coef, clip/clip-param, batch size/num-mini-batch.
- The default of GAE delta is 0.95 and thus the same as in the paper.
- The default for gamma/discount is 0.99 which is also the same as in the paper.

Also not sure if value coeff is the same as value-loss-coef. Previously in the PPO algo it was set to 0.5 as default. The paper does 1.
Next to that I am not sure if num workers is equal to num-steps. num-steps equal to 3 doesn't actually work for 8 workers.

*Use --num-processes 2 --num-steps 16 when low on memory*
Warehouse with FNN
```
python main.py --env-name Warehouse --yaml-file FNN --fnn-hidden-sizes 640,256  --num-processes 8 --num-steps 8 --num-mini-batch 32 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1
```
Warehouse with IAM network:
```
python main.py --env-name Warehouse --yaml-file IAM_static --fnn-hidden-sizes 512,256 --rec-hidden-size 128 --recurrent-policy --num-processes 8 --num-steps 8 --num-mini-batch 32 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1
```

## Additional installation instructions
Preferably you use Pytorch with Cuda enabled but this requires a special version. To check your Cuda version (if installed) execute 
this command in cmd.exe: `nvcc --version`  
You can find the correct command of Pytorch with the specific Cuda version here: https://pytorch.org/  
The command to be executed for Pip packages could look like this (only torch is needed): `pip install torch==1.8.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html`

Install sumo https://sourceforge.net/projects/sumo/files/sumo/version%201.8.0/sumo-win64-1.8.0.msi/download?use_mirror=netix&download=

There is a bug in atari_py. If you already installed it remove by doing `pip uninstall atari_py' and install this version: `pip install -f https://github.com/Kojoley/atari-py/releases atari_py` 