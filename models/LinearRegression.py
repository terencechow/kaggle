# import functools
import tensorflow as tf

from utils import lazy_decorator

# def lazy_decorator(function):
#     attribute = '_cache_' + function.__name__
#
#     @property
#     @functools.wraps(function)
#     def decorator(self):
#         if not hasattr(self, attribute):
#             with tf.variable_scope(function.__name__):
#                 setattr(self, attribute, function(self))
#         return getattr(self, attribute)
#     return decorator


class LinearRegressionModel:
    def __init__(self, X, Y, lr, num_features):
        self.X = X
        self.Y = Y
        self.lr = lr
        self.num_features = num_features
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.error
        self.prediction
        self.optimize

    @lazy_decorator
    def prediction(self):
        w = tf.Variable(tf.truncated_normal(
            [self.num_features, 1]), name='w')
        b = tf.Variable(0.0, name='b')
        return tf.matmul(self.X, w) + b

    @lazy_decorator
    def optimize(self):
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.lr)
        return optimizer.minimize(self.error, global_step=self.global_step)

    @lazy_decorator
    def error(self):
        return tf.reduce_mean(tf.square(self.Y - self.prediction, name='loss'))
