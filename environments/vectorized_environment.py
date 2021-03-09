from environments.worker import Worker
import multiprocessing as mp
import numpy as np
import random

class VectorizedEnvironment(object):
    """
    Creates multiple instances of an environment to run in parallel.
    Each of them contains a separate worker (actor) all of them following
    the same policy
    """

    def __init__(self, parameters, seed):
        print('cpu count', mp.cpu_count())
        if parameters['num_workers'] < mp.cpu_count():
            self.num_workers = parameters['num_workers']
        else:
            self.num_workers = mp.cpu_count()
        # Random seed needs to be set different for each worker (seed + worker_id). Otherwise multiprocessing takes 
        # the current system time, which is the same for all workers!
        self.workers = [Worker(parameters, worker_id, seed + worker_id) for worker_id in range(self.num_workers)]
        self.parameters = parameters
        self.env = parameters['env_type']

    def reset(self):
        """
        Resets each of the environment instances
        """
        for worker in self.workers:
            worker.child.send(('reset', None))
        output = {'obs': [], 'prev_action': []}
        for worker in self.workers:
            obs = worker.child.recv()
            if self.env == 'atari':
                stacked_obs = np.zeros((self.parameters['frame_height'],
                                        self.parameters['frame_width'],
                                        self.parameters['num_frames']))
                stacked_obs[:, :, 0] = obs[:, :, 0]
                obs = stacked_obs
            output['obs'].append(obs)
            output['prev_action'].append(-1)
        return output

    def step(self, actions, prev_stacked_obs):
        """
        Takes an action in each of the enviroment instances
        """
        for worker, action in zip(self.workers, actions):
            worker.child.send(('step', action))
        output = {'obs': [], 'reward': [], 'done': [], 'prev_action': [],
                  'info': []}
        i = 0
        for worker in self.workers:
            obs, reward, done, info = worker.child.recv()
            if self.parameters['flicker']:
                p = 0.5
                prob_flicker = random.uniform(0, 1)
                if prob_flicker > p:
                    obs = np.zeros_like(obs)
            if self.env == 'atari':
                new_stacked_obs = np.zeros((self.parameters['frame_height'],
                                            self.parameters['frame_width'],
                                            self.parameters['num_frames']))
                new_stacked_obs[:, :, 0] = obs[:, :, 0]
                new_stacked_obs[:, :, 1:] = prev_stacked_obs[i][:, :, :-1]
                obs = new_stacked_obs
            output['obs'].append(obs)
            output['reward'].append(reward)
            output['done'].append(done)
            output['info'].append(info)
            i += 1
        output['prev_action'] = actions
        return output

    def action_space(self):
        """
        Returns the dimensions of the environment's action space
        """
        self.workers[0].child.send(('action_space', None))
        action_space = self.workers[0].child.recv()
        return action_space

    def close(self):
        """
        Closes each of the threads in the multiprocess
        """
        for worker in self.workers:
            worker.child.send(('close', None))
