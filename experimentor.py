import os
# import tensorflow as tf
import torch

from a2c_ppo_acktr import algo
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from environments.vectorized_environment import VectorizedEnvironment
import argparse
import yaml
import time
import sacred
from sacred.observers import MongoObserver
import pymongo
from sshtunnel import SSHTunnelForwarder

from main import create_IAM_model
# from old_files.PPO.PPOcontroller import PPOcontroller


class Experimentor(object):
    """
    Creates experimentor object to store interact with the environment and
    the agent and log results.
    """

    def __init__(self, parameters, _run, seed):
        """
        Initializes the experiment by extracting the parameters
        @param parameters a dictionary with many obligatory elements
        <ul>
        <li> "env_type" (SUMO, atari, grid_world),
        <li> algorithm (DQN, PPO)
        <li> maximum_time_steps
        <li> maximum_episode_time
        <li> skip_frames
        <li> save_frequency
        <li> step
        <li> episodes
        and more TODO
        </ul>

        @param logger  TODO what is it exactly? It must have the function
        log_scalar(key, stat_mean, self.step[factor_i])
        """
        self.parameters = parameters
        self.path = self.generate_path(self.parameters)
        self.generate_env(seed)
        self.generate_controller(self.env.action_space(), _run)
        self.train_frequency = self.parameters["train_frequency"]
        # tf.reset_default_graph()

    def generate_path(self, parameters):
        """
        Generate a path to store e.g. logs, models and plots. Check if
        all needed subpaths exist, and if not, create them.
        """
        path = self.parameters['name']
        result_path = os.path.join("results", path)
        model_path = os.path.join("models", path)
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        return path

    def generate_env(self, seed):
        """
        Create environment container that will interact with SUMO
        """
        self.env = VectorizedEnvironment(self.parameters, seed)

    def generate_controller(self, actionmap, run):
        """
        Create controller that will interact with agents
        """
        args = get_args()
        print("Processes: ",args.num_processes)
        if self.parameters['algorithm'] == 'PPO':
            # device = torch.device("cuda:0" if args.cuda else "cpu")
            device = "cpu"
            envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                                 args.gamma, args.log_dir, device, False)
            actor_critic = create_IAM_model(envs, args)
            # self.controller = PPOcontroller(self.parameters, actionmap, run)
            self.controller = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    def print_results(self, info, n_steps=0):
        """
        Prints results to the screen.
        """
        if self.parameters['env_type'] == 'atari':
            print(("Train step {} of {}".format(self.step,
                                                self.maximum_time_steps)))
            print(("-"*30))
            print(("Episode {} ended after {} steps.".format(
                                        self.controller.episodes,
                                        info['l'])))
            print(("- Total reward: {}".format(info['r'])))
            print(("-"*30))
        else:
            print(("Train step {} of {}".format(self.step,
                                                self.maximum_time_steps)))
            print(("-"*30))
            print(("Episode {} ended after {} steps.".format(
                                        self.controller.episodes,
                                        n_steps)))
            print(("- Total reward: {}".format(info)))
            print(("-"*30))

    def run(self):
        """
        Runs the experiment.
        """
        self.maximum_time_steps = int(self.parameters["max_steps"])
        print(self.maximum_time_steps)
        self.step = max(self.parameters["iteration"], 0)
        # reset environment
        step_output = self.env.reset()
        reward = 0
        n_steps = 0
        start = time.time()
        while self.step < self.maximum_time_steps:
            # Select the action to perform
            get_actions_output = self.controller.get_actions(step_output)
            # Increment step
            self.controller.increment_step()
            self.step += 1
            # Get new state and reward given actions a
            next_step_output = self.env.step(get_actions_output['action'],
                                             step_output['obs'])
            if self.parameters['mode'] == 'train':
                # Store experiences in buffer.
                self.controller.add_to_memory(step_output, next_step_output,
                                              get_actions_output)
                # Estimate the returns using value function when time
                # horizon has been reached
                self.controller.bootstrap(next_step_output)
                if self.step % self.train_frequency == 0 and \
                   self.controller.full_memory():
                    self.controller.update()
                step_output = next_step_output
            if self.parameters['env_type'] == 'atari' and 'episode' in next_step_output['info'][0].keys():
                end = time.time()
                # print('Time: ', end - start)
                start = end
                self.print_results(next_step_output['info'][0]['episode'])
            elif self.parameters['env_type'] != 'atari':
                reward += next_step_output['reward'][0]
                n_steps += 1
                if next_step_output['done'][0]:
                    end = time.time()
                    # print('Time: ' , end - start)
                    start = end
                    self.print_results(reward, n_steps)
                    reward = 0
                    n_steps = 0
            self.controller.write_summary()    
            # Tensorflow only stores a limited number of networks.
            self.controller.save_graph()
        self.env.close()

def get_parameters():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument('--config', default=None, help='config file')
    parser.add_argument('--scene', default=None, help='scene')
    parser.add_argument('--flicker', action='store_true', help='flickering game')
    args = parser.parse_args()
    return args

def read_parameters(config_file):
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters['parameters']


def add_mongodb_observer():
    """
    connects the experiment instance to the mongodb database
    """
    # MONGO_HOST = 'TUD-tm2'
    # MONGO_DB = 'influence-aware-memory'
    # PKEY = '~/.ssh/id_rsa'
    # global server
    # try:
    #     print("Trying to connect to mongoDB '{}'".format(MONGO_DB))
    #     server = SSHTunnelForwarder(
    #         MONGO_HOST,
    #         ssh_pkey=PKEY,
    #         remote_bind_address=('127.0.0.1', 27017)
    #         )
    #     server.start()
    #     DB_URI = 'mongodb://localhost:{}/influence-aware-memory'.format(server.local_bind_port)
    #     # pymongo.MongoClient('127.0.0.1', server.local_bind_port)
    #     ex.observers.append(MongoObserver.create(DB_URI, db_name=MONGO_DB, ssl=False))
    #     print("Added MongoDB observer on {}.".format(MONGO_DB))
    # except pymongo.errors.ServerSelectionTimeoutError as e:
    #     print(e)
    print("ONLY FILE STORAGE OBSERVER ADDED")
    from sacred.observers import FileStorageObserver
    ex.observers.append(FileStorageObserver.create('saved_runs'))

ex = sacred.Experiment('influence-aware-memory')
ex.add_config('configs/default.yaml')
add_mongodb_observer()

@ex.automain
def main(parameters, seed, _run):
    exp = Experimentor(parameters, _run, seed)
    exp.run()
