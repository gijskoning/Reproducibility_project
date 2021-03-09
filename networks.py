# import tensorflow as tf

# todo fix Networks class

class Networks(object):
    """
    Creates a network object that can be used to add different neural network
    architectures to the Tensroflow graph. These are: Fully connected,
    convolutional and recurrent neural networks.
    """
    def fcn(input, num_fc_layers, num_fc_units, activation, scope='fcn',
            name='fc_{}'):
        """
        Builds a fully connected neural network
        """
        with tf.variable_scope(scope):
            hidden = input
            for i in range(num_fc_layers):
                hidden = tf.layers.dense(hidden, num_fc_units[i],
                                         activation=activation,
                                         use_bias=False,
                                         name=name.format(i))
        return hidden

    def cnn(input, num_conv_layers, n_filters, kernel_sizes, strides,
            activation, reuse, scope='cnn'):
        """
        Builds a 2D convolutional neural network
        """
        with tf.variable_scope(scope):
            conv = input
            # TODO: add this to parameters
            for i in range(num_conv_layers):
                conv = tf.layers.conv2d(conv, n_filters[i],
                                        kernel_size=kernel_sizes[i],
                                        strides=strides[i],
                                        padding='valid',
                                        activation=activation, reuse=reuse,
                                        name='conv_{}'.format(i))
            output = conv

        return output

    def rnn(input, state_in, num_rec_units, seq_len, scope='rnn'):
        """
        Builds an LSTM recurrent neural network
        """
        # TODO: find out how to generate a multiple layer LSTM network
        with tf.variable_scope(scope):
            s_size = input.get_shape().as_list()[1]
            input = tf.reshape(input, shape=[-1, seq_len, s_size])
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_rec_units)
            output, hidden_state = tf.nn.dynamic_rnn(lstm_cell, input,
                                                     initial_state=state_in,
                                                     dtype=tf.float32,
                                                     sequence_length=seq_len)
            output = tf.reshape(output, [-1, num_rec_units])
        return output, hidden_state
