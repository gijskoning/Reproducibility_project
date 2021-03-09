from controller import Controller
from PPO.PPOmodel import PPOmodel
from buffer import SerialSampling
import numpy as np


class PPOcontroller(Controller):
    """
    Creates PPOController object and can be used to add new experiences to the
    buffer, calculate advantages and returns and update the agent's policy.
    """

    def __init__(self, parameters, action_map, run):
        self.parameters = parameters
        self.num_actions = action_map
        self.model = PPOmodel(self.parameters,
                              self.num_actions)
        self.buffer = SerialSampling(self.parameters,
                                     self.num_actions)
        self.cumulative_rewards = 0
        self.episode_step = 0
        self.episodes = 0
        self.t = 0
        self.stats = {"cumulative_rewards": [],
                      "episode_length": [],
                      "value": [],
                      "learning_rate": [],
                      "entropy": [],
                      "policy_loss": [],
                      "value_loss": []}
        super().__init__(self.parameters, action_map, run)
        if self.parameters['influence']:
            self.seq_len = self.parameters['inf_seq_len']
        elif self.parameters['recurrent']:
            self.seq_len = self.parameters['seq_len']
        else:
            self.seq_len = 1

    def add_to_memory(self, step_output, next_step_output, get_actions_output):
        """
        Append the last transition to buffer and to stats
        """
        self.buffer['obs'].append(step_output['obs'])   
        self.buffer['rewards'].append(next_step_output['reward'])
        self.buffer['dones'].append(next_step_output['done'])
        self.buffer['actions'].append(get_actions_output['action'])
        self.buffer['values'].append(get_actions_output['value'])
        self.buffer['action_probs'].append(get_actions_output['action_probs'])
        # This mask is added so we can ignore experiences added when
        # zero-padding incomplete sequences
        self.buffer['masks'].append([1]*self.parameters['num_workers'])
        self.cumulative_rewards += next_step_output['reward'][0]
        self.episode_step += 1
        self.stats['value'].append(get_actions_output['value'][0])
        self.stats['entropy'].append(get_actions_output['entropy'][0])
        self.stats['learning_rate'].append(get_actions_output['learning_rate'])
        # Note: States out is used when updating the network to feed the
        # initial state of a sequence. In PPO this internal state will not
        # differ that much from the current one. However for DQN we might
        # rather set the initial state as zeros like in Jinke's
        # implementation
        if self.parameters['recurrent']:
            self.buffer['states_in'].append(
                    np.transpose(get_actions_output['state_in'], (1,0,2)))
            self.buffer['prev_actions'].append(step_output['prev_action'])
        if self.parameters['influence']:
            self.buffer['inf_states_in'].append(
                    np.transpose(get_actions_output['inf_state_in'], (1,0,2)))
            self.buffer['inf_prev_actions'].append(step_output['prev_action'])
        if next_step_output['done'][0] and self.parameters['env_type'] != 'atari':
            self.episodes += 1
            self.stats['cumulative_rewards'].append(self.cumulative_rewards)
            self.stats['episode_length'].append(self.episode_step)
            self.cumulative_rewards = 0
            self.episode_step = 0
        if self.parameters['env_type'] == 'atari' and 'episode' in next_step_output['info'][0].keys():
            # The line below is due to live episodes in the openai baselines atari_wrapper
            self.episodes += 1
            self.stats['cumulative_rewards'].append(next_step_output['info'][0]['episode']['r'])
            self.stats['episode_length'].append(self.episode_step)
            self.cumulative_rewards = 0
            self.episode_step = 0
        if self.parameters['recurrent'] or self.parameters['influence']:
            for worker, done in enumerate(next_step_output['done']):
                if done and self.parameters['num_workers'] != 1:
                    # reset worker's internal state
                    self.model.reset_state_in(worker)
                    # zero padding incomplete sequences
                    remainder = len(self.buffer['masks']) % self.seq_len
                    # NOTE: we need to zero-pad all workers to keep the
                    # same buffer dimensions even though only one of them has
                    # reached the end of the episode.
                    if remainder != 0:
                        missing = self.seq_len - remainder
                        self.buffer.zero_padding(missing, worker)
                        self.t += missing
        # NETWORK ARCHITECTURE ANALYSIS
        # if self.parameters['analyze_hidden']:
        #     # write to file
        #     obs_flatten = np.ndarray.flatten(step_output['obs'][0])     
        #     with open('analysis/observation3.txt', 'a') as obs_file:
        #         np.savetxt(obs_file, obs_flatten.reshape(1, obs_flatten.shape[0]), delimiter=',')
        #     hidden_x = get_actions_output['hidden_x'][0]
        #     with open('analysis/hidden_x.txt', 'a') as hidden_file:
        #         np.savetxt(hidden_file, hidden_x.reshape(1, hidden_x.shape[0]), delimiter=',')
        #     hidden_memory = np.concatenate(np.transpose(get_actions_output['inf_state_in'], (1,0,2))[0])
        #     hidden_memory = hidden_memory[:hidden_memory.shape[0]//2]
        #     with open('analysis/hidden_d.txt', 'a') as hidden_file:
        #         np.savetxt(hidden_file, hidden_memory.reshape(1, hidden_memory.shape[0]), delimiter=',')

    def bootstrap(self, next_step_output):
        """
        Computes GAE and returns for a given time horizon
        """
        # TODO: consider the case where the episode is over because the maximum
        # number of steps in an episode has been reached.
        self.t += 1
        if self.t >= self.parameters['time_horizon']:
            evaluate_value_output = self.model.evaluate_value(
                                        next_step_output['obs'],
                                        next_step_output['prev_action'])
            next_value = evaluate_value_output['value']
            batch = self.buffer.get_last_entries(self.t, ['rewards', 'values',
                                                          'dones'])
            advantages = self._compute_advantages(np.array(batch['rewards']),
                                                  np.array(batch['values']),
                                                  np.array(batch['dones']),
                                                  next_value,
                                                  self.parameters['gamma'],
                                                  self.parameters['lambda'])
            self.buffer['advantages'].extend(advantages)
            returns = advantages + batch['values']
            self.buffer['returns'].extend(returns)
            self.t = 0

    def update(self):
        """
        Runs multiple epoch of mini-batch gradient descent to update the model
        using experiences stored in buffer.
        """
        policy_loss = 0
        value_loss = 0
        n_sequences = self.parameters['batch_size'] // self.seq_len
        n_batches = self.parameters['memory_size'] // \
            self.parameters['batch_size']
        import time
        start = time.time()
        for e in range(self.parameters['num_epoch']):
            self.buffer.shuffle()
            for b in range(n_batches):
                batch = self.buffer.sample(b, n_sequences)
                update_model_output = self.model.update_model(batch)
                policy_loss += update_model_output['policy_loss']
                value_loss += update_model_output['value_loss']
        self.buffer.empty()
        self.stats['policy_loss'].append(np.mean(policy_loss))
        self.stats['value_loss'].append(np.mean(value_loss))
        end = time.time()
        print('Time Update: ', end-start)

    def _compute_advantages(self, rewards, values, dones, last_value, gamma,
                            lambd):
        """
        Calculates advantages using genralized advantage estimation (GAE)
        """
        last_advantage = 0
        advantages = np.zeros((self.parameters['time_horizon'],
                               self.parameters['num_workers']),
                              dtype=np.float32)
        for t in reversed(range(self.parameters['time_horizon'])):
            mask = 1.0 - dones[t, :]
            last_value = last_value*mask
            last_advantage = last_advantage*mask
            delta = rewards[t, :] + gamma*last_value - values[t, :]
            last_advantage = delta + gamma*lambd*last_advantage
            advantages[t, :] = last_advantage
            last_value = values[t, :]
        return advantages