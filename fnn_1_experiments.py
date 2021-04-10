from main import Main

import torch

if __name__ == '__main__':
    # Check your computer by calling this in your regular python console: import torch; torch.get_num_threads()
    torch.set_num_threads(16)
    # Running Warehouse GRU 3 times with command:
    args = "--env-name Warehouse --yaml-file FNN --fnn-hidden-sizes 640,256 --algo ppo --use-gae --lr 2.5e-4 --clip-param 0.1 --value-loss-coef 1 --num-processes 16 --num-steps 8 --num-mini-batch 8 --log-interval 100 --use-linear-lr-decay --entropy-coef 0.01"
    arg_list = args.split(" ")
    for i in range(0,3):
        seed_arg = ['--seed', str(10*i)]
        main = Main(arg_list + seed_arg)
        main.run()
