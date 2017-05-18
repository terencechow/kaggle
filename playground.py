import tensorflow as tf
import numpy as np
batch_size = 2
feature_size = 10

y_actuals = np.array([[2], [2]])
features = np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                     [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]])

x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
w = tf.Variable(tf.ones([feature_size, 1]), name='w')
b = tf.Variable(1.0, name='b')
wx = tf.matmul(x, w)
y_predicted = wx + b

loss = tf.reduce_sum(tf.square(y - y_predicted))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    loss_val, y_val = sess.run(
        [loss, y_predicted], feed_dict={x: features, y: y_actuals})
    print('loss: \n{}\nwxb:\n{}'.format(loss_val, y_val))
