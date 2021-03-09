import gym
import logging
from gym import spaces
import os
from .LDM import ldm
from .state_representation import *
import time
from sumolib import checkBinary
import random
from .SumoHelper import SumoHelper
import copy
from .TrafficLightPhases import TrafficLightPhases
from gym.spaces import Box
import numpy as np


class SumoGymAdapter(object):
    """
    An adapter that makes Sumo behave as a proper Gym environment.
    At top level, the actionspace and percepts are in a Dict with the
    trafficPHASES as keys.
    @param maxConnectRetries the max number of retries to connect.
        A retry is needed if the randomly chosen port
        to connect to SUMO is already in use.
    """
    _DEFAULT_PARAMETERS = {'gui':True,  # gui or not
                'scene':'four_grid',  # subdirectory in the aienvs/scenarios/Sumo directory where
                'tlphasesfile':'sample.net.xml',  # file
                'box_bottom_corner':(0, 0),  # bottom left corner of the observable frame
                'box_top_corner':(10, 10),  # top right corner of the observable frame
                'resolutionInPixelsPerMeterX': 1,  # for the observable frame
                'resolutionInPixelsPerMeterY': 1,  # for the observable frame
                'y_t': 6,  # yellow time
                'car_pr': 0.5,  # for automatic route/config generation probability that a car appears
                'car_tm': 2,  #  for automatic route/config generation when the first car appears?
                'route_starts' : [],  #  for automatic route/config generation, ask Rolf
                'route_min_segments' : 0,  #  for automatic route/config generation, ask Rolf
                'route_max_segments' : 0,  #  for automatic route/config generation, ask Rolf
                'route_ends' : [],  #  for automatic route/config generation, ask Rolf
                'generate_conf' : True,  # for automatic route/config generation
                'libsumo' : True,  # whether libsumo is used instead of traci
                'waiting_penalty' : 1,  # penalty for waiting
                'reward_type': 'waiting_time',  # waiting_time or avg_speed
                'lightPositions' : {},  # specify traffic light positions
                'scaling_factor' : 1.0,  # for rescaling the reward? ask Miguel
                'maxConnectRetries':50,  # maximum reattempts to connect by Traci
                'seed': None
                }

    def __init__(self, parameters, seed):
        """
        @param path where results go, like "Experiment ID"
        @param parameters the configuration parameters.
        gui: whether we show a GUI.
        scenario: the path to the scenario to use
        """
        logging.debug(parameters)
        self._parameters = copy.deepcopy(self._DEFAULT_PARAMETERS)
        self._parameters.update(parameters)
        dirname = os.path.dirname(__file__)
        tlPhasesFile = os.path.join(dirname, "scenarios/", self._parameters['scene'], self._parameters['tlphasesfile'])
        self._tlphases = TrafficLightPhases(tlPhasesFile)
        if self._parameters['gui']:
            self._parameters['libsumo'] = False
        self.ldm = ldm(using_libsumo=self._parameters['libsumo'])
        self._takenActions = {}
        self._yellowTimer = {}
        self._chosen_action = None
        self.seed(seed)  # in case no seed is given
        self.original_seed = seed
        self._sumo_helper = SumoHelper(self._parameters, self._seed)
        self._state = LdmMatrixState(self.ldm, [self._parameters['box_bottom_corner'], self._parameters['box_top_corner']], "byCorners")
        self._observation_space = self._compute_observation_space()

    def _compute_observation_space(self):
        self._startSUMO(False)
        _s = self._observe()
        self.frame_height = _s.shape[0]
        self.frame_width = _s.shape[1]
        return Box(low=0, high=1.0, shape=(self.frame_height, self.frame_width), dtype=np.float32)

    def step(self, actions:dict):
        self._set_lights(actions)
        self.ldm.step()
        obs = np.array(self._observe())
        traffic_lights = self.ldm.get_traffic_lights()
        if self._parameters['traffic_lights']:
            obs = np.concatenate((obs[:,8], obs[6,:], traffic_lights), axis=None)
        else:
            obs = np.concatenate((obs[:,8], obs[6,:]), axis=None)
        if self._parameters['num_frames'] > 1:
            obs = np.append(obs, self.prev_obs[:-len(obs)])
            self.prev_obs = np.copy(obs)
        done = self.ldm.isSimulationFinished()
        if self.ldm.SUMO_client.simulation.getTime() >= self._parameters['max_episode_steps']:
            done = True
        global_reward = self._computeGlobalReward()
        # as in openai gym, last one is the info list
        return obs, global_reward, done, []

    def reset(self):
        try:
            logging.debug("LDM closed by resetting")
            self.ldm.close()
        except:
            logging.debug("No LDM to close. Perhaps it's the first instance of training")

        logging.debug("Starting SUMO environment...")
        self._startSUMO()  
        obs = np.array(self._observe())
        traffic_lights = self.ldm.get_traffic_lights()
        if self._parameters['traffic_lights']:
            obs = np.concatenate((obs[:,8], obs[6,:], traffic_lights), axis=None)
        else:
            obs = np.concatenate((obs[:,8], obs[6,:]), axis=None)
        if self._parameters['num_frames'] > 1:
            self.prev_obs = np.zeros(self._parameters['obs_size']-len(obs))
            obs = np.append(obs, self.prev_obs)
            self.prev_obs = np.copy(obs)
        # obs = np.reshape(obs,(self._parameters['frame_width'], self._parameters['frame_height'], 1))
        return obs

        # TODO: change the defaults to something sensible
    def render(self, delay=0.0):
        import colorama
        colorama.init()

        def move_cursor(x, y):
            print ("\x1b[{};{}H".format(y + 1, x + 1))

        def clear():
            print ("\x1b[2J")

        clear()
        move_cursor(100, 100)
        import numpy as np
        np.set_printoptions(linewidth=100)
        time.sleep(delay)

    def seed(self, seed):
        self._seed = seed

    def close(self):
        self.__del__()

    @property
    def observation_space(self):
        # # this is the previous method, which does not take resolution into consideration
        # size = self._state.size()
        # return Box(low=0, high=np.inf, shape=(size[0], size[1]), dtype=np.int32)
        return self._observation_space

    @property
    def action_space(self):
        """
        Returns A gym dict containing the number of action choices for all the
        agents in the environment
        """
        action_space = spaces.Dict({inters:spaces.Discrete(self._tlphases.getNrPhases(inters)) \
                                    for inters in self._tlphases.getIntersectionIds()})
        return action_space['0']

    ########## Private functions ##########################
    def __del__(self):
        logging.debug("LDM closed by destructor")
        if 'ldm' in locals():
            self.ldm.close()

    def _startSUMO(self, gui=None):
        """
        Start the connection with SUMO as a subprocess and initialize
        the traci port, generate route file.
        """
        val = 'sumo'
        if gui is True:
            val = 'sumo-gui'
        elif gui is None:
            val = 'sumo-gui' if self._parameters['gui'] else 'sumo'
        
        maxRetries = self._parameters['maxConnectRetries']
        sumo_binary = checkBinary(val)
        # Try repeatedly to connect
        while True:
            try:
                # this cannot be seeded
                self._port = random.SystemRandom().choice(list(range(10000, 20000)))
                self._sumo_helper._generate_route_file(self._seed)
                conf_file = self._sumo_helper.sumocfg_file
                logging.debug("Configuration: " + str(conf_file))
                self.sumoCmd = [sumo_binary, "-c", conf_file, "-W", "-v", "false",
                           "--default.speeddev", str(self._parameters['speed_dev'])]    
                self.sumoCmd += ["--seed", str(self._seed)]
                self.ldm.start(self.sumoCmd, self._port)
                self._seed += 1
            except Exception as e:
                if str(e) == "connection closed by SUMO" and maxRetries > 0:
                    maxRetries = maxRetries - 1
                    continue
                else:
                    raise
            else:
                break

        self.ldm.init(waitingPenalty=self._parameters['waiting_penalty'], reward_type=self._parameters['reward_type'])  # ignore reward for now
        self.ldm.setResolutionInPixelsPerMeter(self._parameters['resolutionInPixelsPerMeterX'], self._parameters['resolutionInPixelsPerMeterY'])
        self.ldm.setPositionOfTrafficLights(self._parameters['lightPositions'])

        if list(self.ldm.getTrafficLights()) != self._tlphases.getIntersectionIds():
            raise Exception("environment traffic lights do not match those in the tlphasesfile "
                    +self._parameters['tlphasesfile'] + str(self.ldm.getTrafficLights())
                    +str(self._tlphases.getIntersectionIds()))

    def _intToPhaseString(self, intersectionId:str, lightPhaseId: int):
        """
        @param intersectionid the intersection(light) id
        @param lightvalue the PHASES value
        @return the intersection PHASES string eg 'rrGr' or 'GGrG'
        """
        logging.debug("lightPhaseId" + str(lightPhaseId))
        return self._tlphases.getPhase(intersectionId, lightPhaseId)

    def _observe(self):
        """
        Fetches the Sumo state and converts in a proper gym observation.
        The keys of the dict are the intersection IDs (roughly, the trafficLights)
        The values are the state of the TLs
        """
        return self._state.update_state()

    def _computeGlobalReward(self):
        """
        Computes the global reward
        """
        return self._state.update_reward() / self._parameters['scaling_factor']

    def _getActionSpace(self):
        """
        @returns the actionspace: a dict containing <id,phases> where
        id is the intersection id and value is
         all possible actions for each id as specified in tlphases
        """
        return spaces.Dict({inters:spaces.Discrete(self._tlphases.getNrPhases(inters)) \
                            for inters in self._tlphases.getIntersectionIds()})

    def _set_lights(self, actions):
        """
        Take the specified actions in the environment
        @param actions a list of
        """
        intersectionId = '0'
        action = self._intToPhaseString(intersectionId, actions)
            # Retrieve the action that was taken the previous step
        try:
            prev_action = self._takenActions[intersectionId]
        except KeyError:
            # If KeyError, this is the first time any action was taken for this intersection
            prev_action = action
            self._takenActions.update({intersectionId:action})
            self._yellowTimer.update({intersectionId:0})

        # Check if the given action is different from the previous action
        if prev_action != action:
            # Either the this is a true switch or coming grom yellow
            action, self._yellowTimer[intersectionId] = self._correct_action(prev_action, action, self._yellowTimer[intersectionId])

        # Set traffic lights
        self.ldm.setRedYellowGreenState(intersectionId, action)
        self._takenActions[intersectionId] = action

    def _correct_action(self, prev_action, action, timer):

        """
        Check what we are going to do with the given action based on the
        previous action.
        """
        # Check if the agent was in a yellow state the previous step
        if 'y' in prev_action:
            # Check if this agent is in the middle of its yellow state
            if timer > 0:
                new_action = prev_action
                timer -= 1
            # Otherwise we can get out of the yellow state
            else:
                new_action = self._chosen_action
                if not isinstance(new_action, str):
                    raise Exception("chosen action is illegal")
        # We are switching from green to red, initialize the yellow state
        else:
            self._chosen_action = action
            if self._parameters['y_t'] > 0:
                new_action = prev_action.replace('G', 'y')
                timer = self._parameters['y_t'] - 1
            else:
                new_action = action
                timer = 0

        return new_action, timer