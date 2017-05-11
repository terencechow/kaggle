import functools
import tensorflow as tf


def lazy_decorator(function):
    attribute = '_cache_' + function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name__):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)
    return decorator


class LinearRegressionModel:
    def __init__(self, X, Y, lr, num_features):
        self.X = X
        self.Y = Y
        self.lr = lr
        self.num_features = num_features
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.prediction
        self.optimize
        self.error

    @lazy_decorator
    def prediction(self):
        w = tf.Variable(tf.truncated_normal(
            [self.num_features]), name='w')
        b = tf.Variable(0.0, name='b')
        return tf.reduce_sum(w * self.X) + b

    @lazy_decorator
    def optimize(self):
        loss = tf.square(self.Y - self.prediction, name='loss')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr)
        return optimizer.minimize(loss, global_step=self.global_step)

    @lazy_decorator
    def error(self):
        diff = tf.square(self.Y - self.prediction)
        return tf.reduce_mean(diff)
