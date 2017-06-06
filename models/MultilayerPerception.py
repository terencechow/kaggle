import tensorflow as tf
from utils import lazy_decorator
from functools import reduce


class MultilayerPerceptionModel:
    def __init__(self, X, Y, hidden_layer_details, lr, dropout=False):
        self.X = X
        self.Y = Y
        self.hidden_layer_details = hidden_layer_details
        self.dropout = dropout
        self.lr = lr
        self.global_step = tf.Variable(
            0, dtype=tf.int32, trainable=False, name='global_step')
        self.error
        self.prediction
        self.optimize

    @lazy_decorator
    def prediction(self):

        def reduce_hidden_layers(result, y):
            input_size = y["input_size"] if "input_size" in y \
                else result[-1]["tf_var"].get_shape().as_list()[1]
            output_size = y["output_size"]

            result.append({
                "activation": y["activation"],
                "tf_var": tf.Variable(tf.truncated_normal(
                    [input_size, output_size]), name=y["name"],
                    dtype=tf.float32)
            })
            return result

        def reduce_network(result, y):
            current_layer = tf.nn.dropout(result["tf_var"], self.dropout) \
                if self.dropout else result["tf_var"]
            next_layer = y["tf_var"]
            output_size = next_layer.shape[1]
            activate = y["activation"]

            b = tf.Variable(tf.zeros([output_size, ], dtype=tf.float32),
                            name=next_layer.name.split(":", 1)[0] + "_bias")

            output = activate(tf.matmul(current_layer, next_layer) + b)
            return {"tf_var": output}

        layers = reduce(
            reduce_hidden_layers, self.hidden_layer_details, [])
        layers.insert(0, {"tf_var": self.X})
        y_predicted = reduce(reduce_network, layers)["tf_var"]
        return y_predicted

    @lazy_decorator
    def optimize(self):
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.lr)
        return optimizer.minimize(self.error, global_step=self.global_step)

    @lazy_decorator
    def error(self):
        error = tf.reduce_mean(
            tf.square(self.Y - self.prediction, name='loss'))
        tf.summary.scalar('error', error)
        return error
