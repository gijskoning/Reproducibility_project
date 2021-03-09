import argparse
import yaml
import sys
import os
import argparse

def get_config_file():
    dir = os.path.dirname(__file__)
    config_file = os.path.join(dir, 'configs/warehourse_environment.yaml')
    # parser.add_argument('--config', default=config_file, help='config file')
    # args, _ = parser.parse_known_args()
    # config_file = args.config
    return config_file

def read_parameters(scope):
    config_file = get_config_file()
    with open(config_file) as file:
        parameters = yaml.load(file, Loader=yaml.FullLoader)
    return parameters[scope]

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_rows', type=int, default=25,
                        help='number of rows in the warehouse')
    parser.add_argument('--n_columns', type=int, default=25,
                        help='number of columns in the warehouse')
    parser.add_argument('--n_robots_row', type=int, default=6,
                        help='number of robots per row')
    parser.add_argument('--n_robots_column', type=int, default=6,
                        help='number of robots per column')
    parser.add_argument('--distance_between_shelves', type=int, default=4,
                        help='distance between two contiguous shelves')
    parser.add_argument('--robot_domain_size', type=list, default=[5, 5],
                        help='size of the robots domain')
    parser.add_argument('--prob_item_appears', type=int, default=0.025,
                        help='probability of an item appearing at each location')
    parser.add_argument('--learning_robot_id', type=int, default=20,
                        help='learning robot id')
    parser.add_argument('--obs_type', type=str, default='image',
                        help='observation type: image or vector')
    parser.add_argument('--n_steps_episode', type=int, default=100,
                        help='number of steps per episode')
    parser.add_argument('--log_obs', type=bool, default=True,
                        help='wether or not to log the observations')
    parser.add_argument('--log_file', type=str, default='./obs_data.csv',
                        help='path to the log file')

    args = parser.parse_args()

    return args
