import copy
import glob
import os
import time
from collections import deque
from datetime import datetime

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy as PPOPolicy
from a2c_ppo_acktr.storage import RolloutStorage
from environments.warehouse.utils import read_parameters
from evaluation import evaluate
from reproduction_model import IAMPolicy, MLPBase, IAMBase
from plot_data import DataSaver


def create_default_model(envs, args):
    #  Here is the model created! And we should change only this part.
    actor_critic = PPOPolicy(
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        base=None,
        base_kwargs={'recurrent': args.recurrent_policy})
    return actor_critic


def create_IAM_model(envs, args, parameters):
    #  Here is the model created! And we should change only this part.
    base_kwargs = {'recurrent': args.recurrent_policy, 'hidden_sizes': args.fnn_hidden_sizes}
    if args.rec_hidden_size:
        base_kwargs['rnn_hidden_size'] = args.rec_hidden_size
    actor_critic = IAMPolicy(
        obs_shape=envs.observation_space.shape,
        action_space=envs.action_space,
        IAM=parameters['influence'],
        base_kwargs=base_kwargs)
    return actor_critic


class Main:
    def __init__(self, optional_args=None):
        start_time_str = datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
        self.args = get_args(optional_args)
        args = self.args
        assert args.algo == 'ppo'

        self.config_parameters = ""
        if args.env_name == "Warehouse":
            self.config_parameters = read_parameters('parameters', 'warehouse/' + args.yaml_file)
        elif "Breakout" in args.env_name:
            self.config_parameters = read_parameters('parameters', 'atari/' + args.yaml_file)

        self.model_file_name = args.env_name + "_" + start_time_str + ".pt"
        self.data_saver = DataSaver(start_time_str)
        line = "Starting new run: with args " + args.__str__()
        self.data_saver.append(line)
        print(line)
        line = "And parameters: " + self.config_parameters.__str__()
        self.data_saver.append(line)
        print(line)

    def run(self):
        args = self.args
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        print("CUDA is available: ", torch.cuda.is_available())
        if args.cuda:
            print("CUDA enabled")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            if args.cuda_deterministic:
                print("Warning CUDA is requested but is not available")
            else:
                print("CUDA disabled")

        log_dir = os.path.expanduser(args.log_dir)
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)
        print("get_num_thread", torch.get_num_threads())

        device = torch.device("cuda:0" if args.cuda else "cpu")

        envs = make_vec_envs(args.env_name, self.config_parameters, args.seed, args.num_processes,
                             args.gamma, args.log_dir, device, False)

        # actor_critic = create_default_model(envs, args)
        actor_critic = create_IAM_model(envs, args, self.config_parameters)
        actor_critic.to(device)
        if args.algo == 'a2c':
            agent = algo.A2C_ACKTR(
                actor_critic,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                alpha=args.alpha,
                max_grad_norm=args.max_grad_norm)
        # This algorithm should be used for the reproduction project.
        elif args.algo == 'ppo':
            agent = algo.PPO(
                actor_critic,
                args.clip_param,
                args.ppo_epoch,
                args.num_mini_batch,
                args.value_loss_coef,
                args.entropy_coef,
                lr=args.lr,
                eps=args.eps,
                max_grad_norm=args.max_grad_norm)
        elif args.algo == 'acktr':
            agent = algo.A2C_ACKTR(
                actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

        if args.gail:
            assert len(envs.observation_space.shape) == 1
            discr = gail.Discriminator(
                envs.observation_space.shape[0] + envs.action_space.shape[0], 100,
                device)
            file_name = os.path.join(
                args.gail_experts_dir, "trajs_{}.pt".format(
                    args.env_name.split('-')[0].lower()))

            expert_dataset = gail.ExpertDataset(
                file_name, num_trajectories=4, subsample_frequency=20)
            drop_last = len(expert_dataset) > args.gail_batch_size
            gail_train_loader = torch.utils.data.DataLoader(
                dataset=expert_dataset,
                batch_size=args.gail_batch_size,
                shuffle=True,
                drop_last=drop_last)

        rollouts = RolloutStorage(args.num_steps, args.num_processes,
                                  envs.observation_space.shape, envs.action_space,
                                  actor_critic.recurrent_hidden_state_size)

        obs = envs.reset()
        rollouts.obs[0].copy_(obs)
        rollouts.to(device)
        # Always return the average of the last 100 steps. This means the average is sampled.
        episode_rewards = deque(maxlen=100)

        start = time.time()
        num_updates = int(
            args.num_env_steps) // args.num_steps // args.num_processes
        for j in range(num_updates):

            if args.use_linear_lr_decay:
                # decrease learning rate linearly
                utils.update_linear_schedule(
                    agent.optimizer, j, num_updates,
                    agent.optimizer.lr if args.algo == "acktr" else args.lr)

            for step in range(args.num_steps):
                # Sample actions
                with torch.no_grad():
                    value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                        rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                        rollouts.masks[step])

                # Obser reward and next obs
                obs, reward, done, infos = envs.step(action)

                for info in infos:
                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

                # If done then clean the history of observations.
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done])
                bad_masks = torch.FloatTensor(
                    [[0.0] if 'bad_transition' in info.keys() else [1.0]
                     for info in infos])
                rollouts.insert(obs, recurrent_hidden_states, action,
                                action_log_prob, value, reward, masks, bad_masks)

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                    rollouts.masks[-1]).detach()

            if args.gail:
                if j >= 10:
                    envs.venv.eval()

                gail_epoch = args.gail_epoch
                if j < 10:
                    gail_epoch = 100  # Warm up
                for _ in range(gail_epoch):
                    discr.update(gail_train_loader, rollouts,
                                 utils.get_vec_normalize(envs)._obfilt)

                for step in range(args.num_steps):
                    rollouts.rewards[step] = discr.predict_reward(
                        rollouts.obs[step], rollouts.actions[step], args.gamma,
                        rollouts.masks[step])

            rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                     args.gae_lambda, args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = agent.update(rollouts)

            rollouts.after_update()

            # save for every interval-th episode or for the last epoch
            if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
                save_path = os.path.join(args.save_dir, args.algo)
                try:
                    os.makedirs(save_path)
                except OSError:
                    pass

                torch.save([
                    actor_critic,
                    getattr(utils.get_vec_normalize(envs), 'obs_rms', None)
                ], os.path.join(save_path, self.model_file_name))

            if j % args.log_interval == 0 and len(episode_rewards) > 1:
                total_num_steps = (j + 1) * args.num_processes * args.num_steps
                end = time.time()
                elapsed_time = end - start
                data = [j,  # Updates
                        total_num_steps,  # timesteps
                        int(total_num_steps / elapsed_time),  # FPS
                        len(episode_rewards),  # Only useful for print statement
                        np.mean(episode_rewards),  # mean of rewards
                        np.median(episode_rewards),  # median of rewards
                        np.min(episode_rewards),  # min rewards
                        np.max(episode_rewards),  # max rewards
                        dist_entropy,
                        value_loss,
                        action_loss,
                        elapsed_time]
                output = ''.join([str(x) + ',' for x in data])
                self.data_saver.append(output)
                print(
                    f"Updates {data[0]}, num timesteps {data[1]}, FPS {data[2]}, elapsed time {int(data[11])} sec \n Last {data[3]} training episodes: mean/median reward {data[4]:.2f}/{data[5]:.2f}, min/max reward {data[6]:.1f}/{data[7]:.1f}\n")

            if (args.eval_interval is not None and len(episode_rewards) > 1
                    and j % args.eval_interval == 0):
                obs_rms = utils.get_vec_normalize(envs).obs_rms
                evaluate(actor_critic, obs_rms, args.env_name, args.seed,
                         args.num_processes, eval_log_dir, device)


if __name__ == "__main__":
    Main().run()
