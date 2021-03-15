# import tensorflow as tf
# import tensorflow.contrib.layers as c_layers
import os
from networks import Networks as net
import numpy as np


# todo fix model class

class Model(object):
    """
    Creates a model object which can be used by any deep reinforcement learning
    algorithm to configure the neural network architecture of choice.
    """
    def __init__(self, parameters, num_actions):
        self.act_size = num_actions
        self.convolutional = parameters['convolutional']
        self.recurrent = parameters['recurrent']
        self.fully_connected = parameters['fully_connected']
        self.influence = parameters['influence']
        self.learning_rate = parameters['learning_rate']
        self.max_step = parameters['max_steps']
        # self.graph = tf.Graph()
        # self.sess = tf.Session(graph=self.graph)

    def initialize_graph(self):
        """
        Initializes the weights in the Tensorflow graph
        """
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            init = tf.global_variables_initializer()
            self.sess.run(init)

    def save_graph(self, time_step):
        """
        Retrieve network weights to store them.
        """
        with self.graph.as_default():
            file_name = os.path.join('models', self.parameters['name'],
                                     'network')
            print("Saving networks...")
            self.saver.save(self.sess, file_name, time_step)
            print("Saved!")

    def load_graph(self):
        """
        Load pretrained network weights.
        """
        with self.graph.as_default():
            self.saver = tf.train.Saver()
            checkpoint_dir = os.path.join('models', self.parameters['name'])
            ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)

    def build_main_model(self):
        """
        Builds neural network model to approximate policy and value functions
        """
        if self.parameters['obs_type'] == 'image':
            self.observation = tf.placeholder(shape=[None,
                                                    self.parameters["frame_height"],
                                                    self.parameters["frame_width"],
                                                    self.parameters["num_frames"]],
                                              dtype=tf.float32, name='observation')
        else:
            self.observation = tf.placeholder(shape=[None, self.parameters['obs_size']],
                                              dtype=tf.float32, name='observation')
        # normalize input
        if self.parameters['env_type'] == 'atari':
            self.observation = tf.cast(self.observation, tf.float32) / 255.

        if self.convolutional:
            self.feature_vector = net.cnn(self.observation,
                                       self.parameters["num_conv_layers"],
                                       self.parameters["num_filters"],
                                       self.parameters["kernel_sizes"],
                                       self.parameters["strides"],
                                       tf.nn.relu, False, 'cnn')
            network_input = c_layers.flatten(self.feature_vector)
        else:
            self.feature_vector = self.observation
            network_input = self.feature_vector

        if self.fully_connected:
            hidden = net.fcn(network_input, self.parameters["num_fc_layers"],
                             self.parameters["num_fc_units"],
                             tf.nn.relu, 'fcn')

        if self.recurrent:
            self.prev_action = tf.placeholder(shape=[None], dtype=tf.int32,
                                              name='prev_action')
            self.prev_action_onehot = c_layers.one_hot_encoding(self.prev_action,
                                                                self.act_size)
            # network_input = tf.concat([network_input, self.prev_action_onehot], axis=1)

            c_in = tf.placeholder(tf.float32, [None,
                                               self.parameters['num_rec_units']],
                                  name='c_state')
            h_in = tf.placeholder(tf.float32, [None,
                                               self.parameters['num_rec_units']],
                                  name='h_state')
            self.seq_len = tf.placeholder(shape=None, dtype=tf.int32,
                                          name='sequence_length')
            self.state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
            hidden, self.state_out = net.rnn(network_input, self.state_in,
                                             self.parameters['num_rec_units'],
                                             self.seq_len,
                                             'rnn')
        self.hidden = hidden

    def build_influence_model(self):
        """
        Builds influence model
        """
        def manual_dpatch(hidden):
            """
            """
            inf_hidden = []
            if self.parameters['obs_type'] == 'image':
                for predictor in range(self.parameters['inf_num_predictors']):
                    center = np.array(self.parameters['inf_box_center'][predictor])
                    height = self.parameters['inf_box_height'][predictor]
                    width = self.parameters['inf_box_width'][predictor]
                    predictor_hidden = hidden[:, center[0]: center[0] + height,
                                                center[1]: center[1] + width, :]
                    predictor_hidden = c_layers.flatten(predictor_hidden)
                    inf_hidden.append(predictor_hidden)

                inf_hidden = tf.stack(inf_hidden, axis=1)
                hidden_size = inf_hidden.get_shape().as_list()[2]*self.parameters['inf_num_predictors']
                inf_hidden = tf.reshape(inf_hidden, shape=[-1, hidden_size])
            else:
                for predictor in range(self.parameters['inf_num_predictors']):
                    predictor_hidden = hidden[:, self.parameters['dset'][predictor]]
                    inf_hidden.append(predictor_hidden)
                inf_hidden = tf.stack(inf_hidden, axis=1)
            return inf_hidden

        def automatic_dpatch(hidden):
            """
            """
            if self.parameters['obs_type'] == 'image':
                shape = hidden.get_shape().as_list()
                num_regions = shape[1]*shape[2]
                hidden = tf.reshape(hidden, [-1, num_regions, shape[3]])
                inf_hidden = net.fcn(hidden, 1, self.parameters["inf_num_predictors"],
                                     None, 'fcn_auto_dset')
                # for predictor in range(self.parameters['inf_num_predictors']):
                #     name = "weights"+str(predictor)
                    # weights = tf.get_variable(name, shape=(num_regions,1), dtype=tf.dtypes.float32,
                                            #   initializer=tf.ones_initializer, trainable=True)
                    # softmax_weights = tf.contrib.distributions.RelaxedOneHotCategorical(0.1, weights)
                    # softmax_weights = tf.reshape(softmax_weights,[num_regions,1])
                    # softmax_weights = tf.nn.softmax(weights, axis=0)
                    # inf_hidden.append(tf.reduce_sum(softmax_weights*hidden, axis=1))

                # inf_hidden = tf.stack(inf_hidden, axis=1)
                hidden_size = inf_hidden.get_shape().as_list()[2]*self.parameters['inf_num_predictors']
                inf_hidden = tf.reshape(inf_hidden, shape=[-1, hidden_size])
            else:
                inf_hidden = net.fcn(hidden, 1, self.parameters["inf_num_predictors"],
                                     None, 'fcn_auto_dset')
                # shape = hidden.get_shape().as_list()
                # num_variables = shape[1]
                # for predictor in range(self.parameters['inf_num_predictors']):
                #     name = "weights"+str(predictor)
                #     weights = tf.get_variable(name, shape=(1, num_variables), dtype=tf.dtypes.float32,
                #                               initializer=tf.ones_initializer, trainable=True)
                #     softmax_weights = tf.nn.softmax(weights, axis=0)
                #     inf_hidden.append(tf.reduce_sum(softmax_weights*hidden, axis=1))
                # inf_hidden = tf.stack(inf_hidden, axis=1)
            return inf_hidden#, softmax_weights
        
        def attention(hidden_conv, inf_hidden):
            """
            """
            shape = hidden_conv.get_shape().as_list()
            num_regions = shape[1]*shape[2]
            hidden_conv = tf.reshape(hidden_conv, [-1, num_regions, shape[3]])
            inf_hidden_vec = []
            for head in range(self.parameters['num_heads']):
                linear_conv = net.fcn(hidden_conv, 1,
                                        self.parameters['num_att_units'], None,
                                        'att', 'att1_'+str(head))
                linear_hidden = net.fcn(inf_hidden, 1,
                                        self.parameters['num_att_units'], None,
                                        'att', 'att2_'+str(head))
                context = tf.nn.tanh(linear_conv + tf.expand_dims(linear_hidden, 1))
                attention_weights = net.fcn(context, 1, [1], None, 'att', 'att3_'+str(head))
                attention_weights = tf.nn.softmax(attention_weights, axis=1)
                d_patch = tf.reduce_sum(attention_weights*hidden_conv, axis=1)
                inf_hidden_vec.append(tf.concat([d_patch, tf.reshape(attention_weights, shape=[-1, num_regions])], axis=1))
            inf_hidden = tf.concat(inf_hidden_vec, axis=1)
            return inf_hidden

        def unroll(iter, state, hidden_states):#, softmax_weights):
            """
            """
            hidden = tf.cond(self.update_bool,
                                  lambda: tf.gather_nd(self.feature_vector,
                                                       self.indices+iter),
                                  lambda: self.feature_vector)
            inf_prev_action = tf.cond(self.update_bool,
                                      lambda: tf.gather_nd(self.inf_prev_action,
                                                           self.indices+iter),
                                      lambda: self.inf_prev_action)
            inf_hidden = state.h
            if self.parameters['attention']:
                inf_hidden = attention(hidden, inf_hidden)
            elif self.parameters['automatic_dpatch']:
                inf_hidden = automatic_dpatch(hidden)
            else:
                inf_hidden = manual_dpatch(hidden)


            inf_prev_action_onehot = c_layers.one_hot_encoding(inf_prev_action,
                                                               self.act_size)
            # inf_hidden = tf.concat([inf_hidden, inf_prev_action_onehot], axis=1)
            inf_hidden, state = net.rnn(inf_hidden, state,
                                        self.parameters['inf_num_rec_units'],
                                        self.inf_seq_len, 'inf_rnn')
            hidden_states = hidden_states.write(iter, inf_hidden)
            iter += 1

            return [iter, state, hidden_states]#, softmax_weights]

        def condition(iter, state, hidden_states):#, softmax_weights):
            return tf.less(iter, self.n_iterations)

        inf_c = tf.placeholder(tf.float32,
                               [None, self.parameters['inf_num_rec_units']],
                               name='inf_c_state')
        inf_h = tf.placeholder(tf.float32,
                               [None, self.parameters['inf_num_rec_units']],
                               name='inf_h_state')
        self.inf_state_in = tf.contrib.rnn.LSTMStateTuple(inf_c, inf_h)
        self.inf_seq_len = tf.placeholder(tf.int32, None,
                                          name='inf_sequence_length')
        self.n_iterations = tf.placeholder(tf.int32, None,
                                          name='n_iterations')
        self.inf_prev_action = tf.placeholder(shape=[None], dtype=tf.int32,
                                              name='inf_prev_action')
        size = self.parameters['batch_size']*self.parameters['num_workers']
        self.indices = np.arange(0, size, self.parameters['inf_seq_len'])
        self.indices = tf.constant(np.reshape(self.indices, [-1, 1]),
                                   dtype=tf.int32)
        self.update_bool = tf.placeholder(tf.bool, [], name='update_bool')
        # outputs of the loop cant change size, thus we need to initialize
        # the hidden states vector and overwrite with new values
        hidden_states = tf.TensorArray(dtype=tf.float32, size=self.n_iterations)
        # softmax_weights = np.zeros((49, 1), dtype='f')
        # Unroll the RNN to fetch intermediate internal states and compute
        # attention weights
        _, self.inf_state_out, hidden_states = tf.while_loop(condition,
                                                             unroll,
                                                             [0,
                                                              self.inf_state_in,
                                                              hidden_states])
                                                            #   softmax_weights])
        self.inf_hidden = hidden_states.stack()
        self.inf_hidden = tf.reshape(tf.transpose(self.inf_hidden,
                                                  perm=[1,0,2]),
                                     [-1, self.parameters['inf_num_rec_units']])

    def build_optimizer(self):
        """
        """
        raise NotImplementedError

    def evaluate_policy(self):
        """
        """
        raise NotImplementedError

    def evaluate_value(self):
        """
        """
        raise NotImplementedError

    def increment_step(self):
        """
        """
        self.sess.run(self.increment)

    def get_current_step(self):
        """
        """
        return self.sess.run(self.step)
