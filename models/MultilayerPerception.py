import tensorflow as tf
from utils import lazy_decorator


class MultilayerPerceptionModel:
    def __init__(self, X, Y, input_size, num_hidden_layers, num_neurons,
                 learning_rate=None,
                 regularization=0.01, activation=tf.nn.relu, dropout=False):
        self.X = X
        self.Y = Y
        self.activation = activation
        self.regularization = regularization
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons = num_neurons
        self.input_size = input_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.weights
        self.prediction
        if learning_rate is not None:
            self.error
            self.optimize

    @lazy_decorator
    def weights(self):
        layers = []

        if self.num_hidden_layers == 0:
            layer_1_output_size = 1
        else:
            layer_1_output_size = self.num_neurons

        layers.append(tf.Variable(tf.truncated_normal([
            self.input_size, layer_1_output_size]),
            name="w1", dtype=tf.float32))

        for i in range(self.num_hidden_layers):
            if i == self.num_hidden_layers - 1:
                output_size = 1
            else:
                output_size = self.num_neurons

            layers.append(tf.Variable(
                tf.truncated_normal([self.num_neurons, output_size]),
                name="w{}".format(i + 2), dtype=tf.float32))
        return layers

    @lazy_decorator
    def prediction(self):
        next_input = self.X
        for i, current_layer in enumerate(self.weights):

            if self.dropout:
                next_input = tf.nn.dropout(next_input, self.dropout)

            b = tf.Variable(tf.zeros([current_layer.shape[1], ],
                                     dtype=tf.float32),
                            name=current_layer.name.split(":", 1)[0] + "_bias")

            next_input = tf.matmul(next_input, current_layer) + b
            if i < len(self.weights) - 1:
                next_input = self.activation(next_input)
        return next_input

    @lazy_decorator
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        return optimizer.minimize(self.error, global_step=self.global_step)

    @lazy_decorator
    def error(self):
        error = tf.reduce_mean(
            tf.square(self.Y - self.prediction, name='loss'))
        tf.summary.scalar('Error', error)

        l2_loss = tf.reduce_sum(
            [tf.nn.l2_loss(layer_weights) for layer_weights in self.weights]) \
            * self.regularization
        error = error + l2_loss
        return error
