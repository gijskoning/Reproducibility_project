# import tensorflow as tf
# import tensorflow.contrib.layers as c_layers
from old_files.model import Model
import numpy as np
import time

# todo fix PPOModel class.

class PPOmodel(Model):
    """
    Creates a PPOmodel object which builds a graph in Tensorflow that consists
    of the neural network model and the set of operations to optimize the
    network using Proximal Policy Optimization https://arxiv.org/abs/1707.06347.
    It also contains methods that call Tensorflow for inference and to update
    the model.
    """

    def __init__(self, parameters, num_actions):
        super().__init__(parameters, num_actions)
        self.parameters = parameters
        self.beta = parameters['beta']
        self.epsilon = parameters['epsilon']
        with self.graph.as_default():
            self.step = tf.Variable(0, name="global_step",
                                    trainable=False, dtype=tf.int32)
            self.increment = tf.assign(self.step,
                                       tf.add(self.step, 1))
            self.build_main_model()
            if self.influence:
                self.build_influence_model()
            self.build_actor_critic()
            self.build_ppo_optimizer()
        self.reset_state_in()
        if self.parameters['load']:
            self.load_graph()
        else:
            self.initialize_graph()
        self.forward_pass_times = []
        self.backward_pass_times = []

    def build_actor_critic(self):
        """
        Adds actor and critic heads to Tensorflow graph
        """
        if self.influence:
            hidden = tf.concat([self.hidden, self.inf_hidden], axis=1)
        else:
            hidden = self.hidden

        self.logits = tf.layers.dense(hidden, self.act_size,
                                      activation=None, use_bias=False,
                                      kernel_initializer=
                                      c_layers.variance_scaling_initializer(
                                       factor=0.01))

        self.action_probs = tf.nn.softmax(self.logits, name="action_probs")
        self.action = tf.reduce_sum(tf.multinomial(self.logits, 1), axis=1)
        self.action = tf.identity(self.action, name="action")

        self.value = tf.reduce_sum(tf.layers.dense(hidden, 1, activation=None),
                                   axis=1)
        self.value = tf.identity(self.value, name="value_estimate")

        self.entropy = -tf.reduce_sum(self.action_probs
                                      * tf.log(self.action_probs + 1e-10),
                                      axis=1)
        self.action_holder = tf.placeholder(shape=[None], dtype=tf.int32,
                                            name='action_holder')
        self.actions_onehot = c_layers.one_hot_encoding(self.action_holder,
                                                        self.act_size)

        self.old_action_probs = tf.placeholder(shape=[None, self.act_size],
                                               dtype=tf.float32,
                                               name='old_probs')

        self.action_prob = tf.reduce_sum(self.action_probs*self.actions_onehot,
                                         axis=1)
        self.old_action_prob = tf.reduce_sum(self.old_action_probs
                                             * self.actions_onehot, axis=1)

    def build_ppo_optimizer(self):
        """
        Adds optimization operations to Tensorflow graph
        """
        decay_epsilon = tf.train.polynomial_decay(self.parameters["epsilon"],
                                                  self.step,
                                                  self.parameters["max_steps"],
                                                  1e-5, power=1.0)
        # Ignore sequence padding
        self.mask_input = tf.placeholder(shape=[None], dtype=tf.float32,
                                         name='masks')
        self.masks = tf.cast(self.mask_input, tf.int32)
        # Value function optimizer
        self.returns = tf.placeholder(shape=[None], dtype=tf.float32,
                                      name='return')
        self.old_values = tf.placeholder(shape=[None], dtype=tf.float32,
                                         name='old_values')

        clipped_value_estimate = self.old_values + \
            tf.clip_by_value(self.value - self.old_values, -decay_epsilon,
                             decay_epsilon)
        v1 = tf.squared_difference(self.returns, self.value)
        v2 = tf.squared_difference(self.returns, clipped_value_estimate)
        self.value_loss = tf.reduce_mean(tf.dynamic_partition(tf.maximum(v1, v2),
                                                              self.masks, 2)[1])
        # Policy optimizer
        self.advantages = tf.placeholder(shape=[None], dtype=tf.float32,
                                         name='advantages')
        importance = self.action_prob / (self.old_action_prob + 1e-10)
        p1 = importance * self.advantages
        p2 = tf.clip_by_value(importance, 1.0 - decay_epsilon,
                              1.0 + decay_epsilon) * self.advantages

        self.policy_loss = -tf.reduce_mean(tf.dynamic_partition(tf.minimum(p1, p2),
                                                                self.masks, 2)[1])

        # Entropy bonus
        self.entropy_bonus = tf.reduce_mean(tf.dynamic_partition(self.entropy,
                                                                 self.masks, 2)[1])
        decay_beta = tf.train.polynomial_decay(self.parameters["beta"],
                                               self.step,
                                               self.parameters["max_steps"],
                                               1e-2, power=1.0)
        # Loss function
        self.loss = self.policy_loss + self.parameters['c1']*self.value_loss - \
            decay_beta*self.entropy_bonus

        self.learning_rate = tf.train.polynomial_decay(self.parameters["learning_rate"],
                                                       self.step,
                                                       self.parameters["max_steps"],
                                                       1e-10, power=1.0)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.update = optimizer.minimize(self.loss)

    def evaluate_policy(self, observation, prev_action):
        """
        Evaluates policy given current observation and previous action
        """
        feed_dict = {self.observation: observation}
        run_dict = {'action': self.action, 'value': self.value,
                    'action_probs': self.action_probs, 'entropy': self.entropy,
                    'learning_rate': self.learning_rate, 'hidden_x': self.hidden}
        if self.recurrent:
            feed_dict[self.state_in] = self.state_in_value
            feed_dict[self.seq_len] = 1
            feed_dict[self.prev_action] = prev_action
            run_dict['state_out'] = self.state_out
        if self.influence:
            feed_dict[self.inf_state_in] = self.inf_state_in_value
            feed_dict[self.inf_seq_len] = 1
            feed_dict[self.n_iterations] = 1
            feed_dict[self.inf_prev_action] = prev_action
            feed_dict[self.update_bool] = False
            run_dict['inf_state_out'] = self.inf_state_out
        start = time.time()
        output_list = self.sess.run(list(run_dict.values()),
                                    feed_dict=feed_dict)
        end = time.time()
        self.forward_pass_times.append(end-start)
        output_dict = dict(zip(list(run_dict.keys()), output_list))
        if self.recurrent:
            output_dict['state_in'] = self.state_in_value
            self.state_in_value = output_dict['state_out']
        if self.influence:
            output_dict['inf_state_in'] = self.inf_state_in_value
            self.inf_state_in_value = output_dict['inf_state_out']
        return output_dict

    def evaluate_value(self, observation, prev_action):
        """
        Evaluates value given current observation and previous action
        """
        feed_dict = {self.observation: observation}
        run_dict = {'value': self.value}
        if self.recurrent:
            feed_dict[self.state_in] = self.state_in_value
            feed_dict[self.seq_len] = 1
            feed_dict[self.prev_action] = prev_action
        if self.influence:
            feed_dict[self.inf_state_in] = self.inf_state_in_value
            feed_dict[self.inf_seq_len] = 1
            feed_dict[self.n_iterations] = 1
            feed_dict[self.inf_prev_action] = prev_action
            feed_dict[self.update_bool] = False
        output_list = self.sess.run(list(run_dict.values()),
                                    feed_dict=feed_dict)
        output_dict = dict(zip(list(run_dict.keys()), output_list))
        return output_dict

    def update_model(self, batch):
        """
        Updates model using experiences stored in buffer
        """
        if self.parameters['obs_type'] == 'image':
            obs = np.reshape(batch['obs'], [-1, self.parameters['frame_height'],
                                            self.parameters['frame_width'],
                                            self.parameters['num_frames']])
        else:
            obs = np.reshape(batch['obs'], [-1, self.parameters['obs_size']])
        feed_dict = {self.observation: obs,
                     self.returns: np.reshape(batch['returns'], -1),
                     self.old_values: np.reshape(batch['values'], -1),
                     self.old_action_probs: np.reshape(batch['action_probs'],
                                                       [-1, self.act_size]),
                     self.advantages: np.reshape(batch['advantages'], -1),
                     self.action_holder: np.reshape(batch['actions'], -1),
                     self.mask_input: np.reshape(batch['masks'], -1)}
        if self.recurrent:
            start_sequence_idx = np.arange(0, np.shape(batch['states_in'])[1],
                                           self.parameters['seq_len'])
            state_in = np.array(batch['states_in'])[:, start_sequence_idx, :, :]
            c_in = np.reshape(state_in[:, :, 0, :],
                              [-1, self.parameters['num_rec_units']])
            h_in = np.reshape(state_in[:, :, 1, :],
                              [-1, self.parameters['num_rec_units']])
            state_in = (c_in, h_in)
            feed_dict[self.state_in] = state_in
            feed_dict[self.seq_len] = self.parameters['seq_len']
            feed_dict[self.prev_action] = np.reshape(batch['prev_actions'], -1)
        if self.influence:
            start_sequence_idx = np.arange(0, np.shape(batch['inf_states_in'])[1],
                                           self.parameters['inf_seq_len'])
            state_in = np.array(batch['inf_states_in'])[:,
                                                        start_sequence_idx, :, :]
            c_in = np.reshape(state_in[:, :, 0, :],
                              [-1, self.parameters['inf_num_rec_units']])
            h_in = np.reshape(state_in[:, :, 1, :],
                              [-1, self.parameters['inf_num_rec_units']])
            inf_state_in = (c_in, h_in)
            feed_dict[self.inf_state_in] = inf_state_in
            feed_dict[self.inf_seq_len] = 1
            feed_dict[self.n_iterations] = self.parameters['inf_seq_len']
            feed_dict[self.inf_prev_action] = np.reshape(batch['inf_prev_actions'], -1)
            feed_dict[self.update_bool] = True
        run_dict = {'value_loss': self.value_loss,
                    'policy_loss': self.policy_loss,
                    'loss': self.loss,
                    'update': self.update,
                    'learning_rate': self.learning_rate}
                    # 'softmax_weights': self.softmax_weights}
        start = time.time()
        output_list = self.sess.run(list(run_dict.values()),
                                    feed_dict=feed_dict)
        output_dict = dict(zip(list(run_dict.keys()), output_list))
        end = time.time()
        self.backward_pass_times.append(end-start)
        # breakpoint()
        # print(np.max(output_dict['softmax_weights']))
        # print(np.argmax(output_dict['softmax_weights']))
        return output_dict

    def reset_state_in(self, worker=None):
        """
        Initialize internal state of recurrent networks to zero
        """
        if worker is None:
            if self.recurrent:
                c_init = np.zeros((self.parameters['num_workers'],
                                   self.parameters['num_rec_units']),
                                  np.float32)
                h_init = np.zeros((self.parameters['num_workers'],
                                   self.parameters['num_rec_units']),
                                  np.float32)
                self.state_in_value = (c_init, h_init)
            if self.influence:
                c_init = np.zeros((self.parameters['num_workers'],
                                   self.parameters['inf_num_rec_units']),
                                  np.float32)
                h_init = np.zeros((self.parameters['num_workers'],
                                   self.parameters['inf_num_rec_units']),
                                  np.float32)
                self.inf_state_in_value = (c_init, h_init)
        else:
            if self.recurrent:
                c_init = np.zeros(self.parameters['num_rec_units'],
                                  np.float32)
                h_init = np.zeros(self.parameters['num_rec_units'],
                                  np.float32)
                self.state_in_value[0][worker] = c_init
                self.state_in_value[1][worker] = h_init
            if self.influence:
                c_init = np.zeros(self.parameters['inf_num_rec_units'],
                                  np.float32)
                h_init = np.zeros(self.parameters['inf_num_rec_units'],
                                  np.float32)
                self.inf_state_in_value[0][worker] = c_init
                self.inf_state_in_value[1][worker] = h_init
