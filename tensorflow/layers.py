import tensorflow as tf

from summary_utils import variable_summaries

def activate(x, type):
    '''
    Activation Function Selector
    '''
    type = type.lower()
    return {'relu': tf.nn.relu(x, name='relu'),
            'linear': x,
            'none':x,
            }.get(type, x)

def conv2d(name, input, shape,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        strides=[1,1,1,1],
        padding='SAME',
        bias=True,
        activation='relu'):
    '''
    2D convolution Layer with smart variable reuse.
    '''

    def conv2d_helper(input, shape, dtype, initializer, strides, padding, activation):
        kernel = tf.get_variable('weights', shape=shape, dtype=dtype,
                initializer=initializer)
        variable_summaries(kernel, 'kernels')
        conv = tf.nn.conv2d(input, kernel, strides, padding=padding)

        if bias:
            biases = tf.get_variable('biases', shape=shape[-1], dtype=dtype,
                    initializer=tf.constant_initializer(0.0))
            variable_summaries(biases, 'biases')
            conv = tf.nn.bias_add(conv, biases)

        output = activate(conv, type=activation)
        return output, kernel, biases

    with tf.variable_scope(name) as scope:
        try:
            return conv2d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)
        except ValueError:
            scope.reuse_variables()
            return conv2d_helper(input, shape, dtype, initializer,
                    strides, padding, activation)

def batch_norm(x, phase, name='bn'):
    """
    Batch normalization on convolutional maps.
    Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
    Args:
        x:           Tensor, 4D BHWD input maps; or 5D BTHWD
        phase:       boolean tf.Variabe, true indicates training phase
        name:        string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.variable_scope(name):
        x_shape = x.get_shape().as_list()
        ndims = len(x_shape)
        n_out = x_shape[-1]
        beta = tf.get_variable('beta', shape=[n_out], dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        gamma = tf.get_variable('gamma', shape=[n_out], dtype=tf.float32,
                initializer=tf.constant_initializer(1.0))

        if ndims==4:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
        elif ndims==5:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2,3], name='moments')

        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(phase,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
    return normed

def fc(name, input, units,
        dtype=tf.float32,
        initializer=tf.contrib.layers.xavier_initializer(),
        activation='relu'):
    '''
    Input must be 2D (batch, features)
    Applys a fully connected layer
    '''
    def fc_helper(input, units, initializer, activation):
        # This must be a non tensor value present at graph creation
        # Used to determine 'Weights' first dimension
        fan_in = input.get_shape()[1]

        weights = tf.get_variable('weights', shape=(fan_in, units), dtype=dtype, initializer=initializer)
        biases = tf.get_variable('biases', shape=(units), dtype=dtype,
                initializer=tf.constant_initializer(0.0))

        variable_summaries(weights, 'weights')
        variable_summaries(biases, 'biases')

        pre_activate = tf.nn.bias_add(tf.matmul(input, weights), biases)
        output = activate(pre_activate, type=activation)
        return output, weights, biases

    with tf.variable_scope(name) as scope:
        try:
            return fc_helper(input, units, initializer, activation)
        except ValueError:
            scope.reuse_variables()
            return fc_helper(input, units, initializer, activation)
