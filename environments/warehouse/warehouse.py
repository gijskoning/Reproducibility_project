from environments.warehouse.item import Item
from environments.warehouse.robot import Robot
from environments.warehouse.utils import *
import numpy as np
import copy
import random
from gym import spaces
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import csv


class Warehouse(object):
    """
    warehouse environment
    """

    ACTIONS = {0: 'UP',
               1: 'DOWN',
               2: 'LEFT',
               3: 'RIGHT'}

    def __init__(self, parameters):
        self.n_columns = 7
        self.n_rows = 7
        self.n_robots_row = 1
        self.n_robots_column = 1
        self.distance_between_shelves = 6
        self.robot_domain_size = [7, 7]
        self.prob_item_appears = 0.05
        # The learning robot
        self.learning_robot_id = 0
        self.max_episode_length = 100
        self.render_bool = False
        self.render_delay = 0.5
        self.obs_type = 'vector'
        self.items = []
        self.img = None

        # todo temp for gym adaptation
        self.reward_range = 1
        self.metadata = 2
        self.observation_minmax = np.zeros(2)

        # self.reset()
        self.max_waiting_time = 8
        self.total_steps = 0
        self.parameters = parameters
        self.reset()
        # self.seed(seed)

    ############################## Override ###############################

    def reset(self):
        """
        Resets the environment's state
        """
        self.robot_id = 0
        self._place_robots()
        self.item_id = 0
        self.items = []
        self._add_items()
        obs = self._get_observation()
        if self.parameters['num_frames'] > 1:
            self.prev_obs = np.zeros(self._get_obs_size() - len(obs))
            obs = np.append(obs, self.prev_obs)
            self.prev_obs = np.copy(obs)
        self.episode_length = 0
        return obs

    def step(self, action):
        """
        Performs a single step in the environment.
        """
        self._robots_act([action])
        self._increase_item_waiting_time()
        reward = self._compute_reward(self.robots[self.learning_robot_id])
        self._remove_items()
        self._add_items()
        obs = self._get_observation()
        if self.parameters['num_frames'] > 1:
            obs = np.append(obs, self.prev_obs[:-len(obs)])
            self.prev_obs = np.copy(obs)
        # Check whether learning robot is done
        # done = self.robots[self.learning_robot_id].done
        self.total_steps += 1
        self.episode_length += 1
        done = (self.max_episode_length <= self.episode_length)
        if self.render_bool:
            self.render(self.render_delay)

        # PPO algo expects "obs, reward, done, infos" here
        # Todo not sure if r should be equal to reward
        infos = {'episode': {'r': reward}}
        return obs, reward, done, infos

    def _get_obs_size(self):
        return self.observation_space.shape[0]

    @property
    def observation_space(self):
        """
        From the paper:
        The observations are a combination
        of the agent’s location (one-hot encoded vector) and the 24
        item binary variables. In the experiments where d-sets are
        manually selected, the RNN in IAM only receives the latter
        variables while the FNN processes the entire vector.
        """
        # todo need to fix observation space
        # The observation space consists\
        # PPO algorithm
        # space = spaces.MultiBinary([self.parameters['num_frames']*(49+24)])
        # todo not sure if agent and item space are in correct order here
        # agent_position_space = spaces.Box(0, 1, (self.parameters['num_frames'] * (49),), dtype=int)
        # item_space = spaces.Box(0, 9, (self.parameters['num_frames'] * (24),), dtype=int)
        # return flatten_space(space)
        # for _ in range(self.parameters['num_frames']):
        # flatten = spaces.flatten_space(
        #     spaces.Dict({'position_one_hot_vector': agent_position_space, 'item_space': item_space}))
        # different way of creating the observation space
        flatten = spaces.Box(0, 1, (self.parameters['num_frames'] * (49+24),), dtype=int)

        return flatten

    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        # todo environment has different agents??
        #  Currently there is only one robot setup. So simplify the current code:
        n_actions = spaces.Discrete(len(self.ACTIONS))
        # action_dict = {robot.get_id: n_actions for robot in self.robots}
        # action_space = spaces.Dict(action_dict)
        # action_space.n = 4
        action_space = n_actions
        return action_space

    def render(self, delay=0.0):
        """
        Renders the environment
        """
        bitmap = self._get_state()
        position = self.robots[self.learning_robot_id].get_position
        bitmap[position[0], position[1], 1] += 1
        im = bitmap[:, :, 0] - 2 * bitmap[:, :, 1]
        if self.img is None:
            fig, ax = plt.subplots(1)
            self.img = ax.imshow(im, vmin=-2, vmax=1)
            for robot_id, robot in enumerate(self.robots):
                domain = robot.get_domain
                y = domain[0]
                x = domain[1]
                color = 'k'
                linestyle = '-'
                linewidth = 2
                rect1 = patches.Rectangle((x + 0.5, y + 0.5), self.robot_domain_size[0] - 2,
                                          self.robot_domain_size[1] - 2, linewidth=linewidth,
                                          edgecolor=color, linestyle=linestyle,
                                          facecolor='none')
                rect2 = patches.Rectangle((x - 0.48, y - 0.48), self.robot_domain_size[0] - 0.02,
                                          self.robot_domain_size[1] - 0.02, linewidth=3,
                                          edgecolor=color, linestyle=linestyle,
                                          facecolor='none')
                self.img.axes.get_xaxis().set_visible(False)
                self.img.axes.get_yaxis().set_visible(False)
                ax.add_patch(rect1)
                ax.add_patch(rect2)
        else:
            self.img.set_data(im)
        plt.pause(delay)
        plt.draw()

    def close(self):
        pass

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

    ######################### Private Functions ###########################

    def _place_robots(self):
        """
        Sets robots initial position at the begining of every episode
        """
        self.robots = []
        domain_rows = np.arange(0, self.n_rows, self.robot_domain_size[0] - 1)
        domain_columns = np.arange(0, self.n_columns, self.robot_domain_size[1] - 1)
        for i in range(self.n_robots_row):
            for j in range(self.n_robots_column):
                robot_domain = [domain_rows[i], domain_columns[j],
                                domain_rows[i + 1], domain_columns[j + 1]]
                robot_position = [robot_domain[0] + self.robot_domain_size[0] // 2,
                                  robot_domain[1] + self.robot_domain_size[1] // 2]
                self.robots.append(Robot(self.robot_id, robot_position,
                                         robot_domain))
                self.robot_id += 1

    def _add_items(self):
        """
        Add new items to the designated locations in the environment which
        need to be collected by the robots
        """
        item_columns = np.arange(0, self.n_columns)
        item_rows = np.arange(0, self.n_rows, self.distance_between_shelves)
        item_locs = None
        if len(self.items) > 0:
            item_locs = [item.get_position for item in self.items]
        for row in item_rows:
            for column in item_columns:
                loc = [row, column]
                loc_free = True
                if item_locs is not None:
                    loc_free = loc not in item_locs
                if np.random.uniform() < self.prob_item_appears and loc_free:
                    self.items.append(Item(self.item_id, loc))
                    self.item_id += 1
        item_rows = np.arange(0, self.n_rows)
        item_columns = np.arange(0, self.n_columns, self.distance_between_shelves)
        if len(self.items) > 0:
            item_locs = [item.get_position for item in self.items]
        for row in item_rows:
            for column in item_columns:
                loc = [row, column]
                loc_free = True
                if item_locs is not None:
                    loc_free = loc not in item_locs
                if np.random.uniform() < self.prob_item_appears and loc_free:
                    self.items.append(Item(self.item_id, loc))
                    self.item_id += 1

    def _get_state(self):
        """
        Generates a 3D bitmap: First layer shows the location of every item.
        Second layer shows the location of the robots.
        """
        state_bitmap = np.zeros([self.n_rows, self.n_columns, 2], dtype=np.int)
        for item in self.items:
            item_pos = item.get_position
            state_bitmap[item_pos[0], item_pos[1], 0] = 1
        for robot in self.robots:
            robot_pos = robot.get_position
            state_bitmap[robot_pos[0], robot_pos[1], 1] = 1
        return state_bitmap

    def _get_observation(self):
        """
        Generates the individual observation for every robot given the current
        state and the robot's designated domain.
        """
        state = self._get_state()
        observation = self.robots[self.learning_robot_id].observe(state, self.obs_type)
        # self.observation_minmax[0] = np.maximum(observation.max(), self.observation_minmax[1])
        # self.observation_minmax[1] = np.minimum(observation.min(), self.observation_minmax[0])
        # print("self.observation_minmax", self.observation_minmax)
        # print("observation", observation)S
        return observation

    def _robots_act(self, actions):
        """
        All robots take an action in the environment.
        """
        for action, robot in zip(actions, self.robots):
            robot.act(action)

    def _compute_reward(self, robot):
        """
        Computes reward for the learning robot.
        """
        reward = 0
        robot_pos = robot.get_position
        robot_domain = robot.get_domain
        for item in self.items:
            item_pos = item.get_position
            if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                reward += 1
        return reward

    def _remove_items(self):
        """
        Removes items collected by robots. Robots collect items by steping on
        them
        """
        for robot in self.robots:
            robot_pos = robot.get_position
            for item in self.items:
                item_pos = item.get_position
                if robot_pos[0] == item_pos[0] and robot_pos[1] == item_pos[1]:
                    self.items.remove(item)
                elif item.get_waiting_time >= self.max_waiting_time:
                    self.items.remove(item)

    def _increase_item_waiting_time(self):
        """
        Increases items waiting time
        """
        for item in self.items:
            item.increase_waiting_time()
