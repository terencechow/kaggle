import tensorflow as tf
from matplotlib import pyplot as plt

logs_dir = '../logs'
# get some data

x = [6.2, 9.5, 10.5, 7.7, 8.6, 34.1, 11, 6.9, 7.3, 15.1, 29.1, 2.2, 5.7, 2,
     2.5, 4, 5.4, 2.2, 7.2, 15.1, 16.5, 18.4, 36.2, 39.7, 18.5, 23.3, 12.2,
     5.6, 21.8, 21.6, 9, 3.6, 5, 28.6, 17.4, 11.3, 3.4, 11.9, 10.5, 10.7, 10.8,
     4.8]
y = [29, 44, 36, 37, 53, 68, 75, 18, 31, 25, 34, 14, 11, 11, 22, 16, 27, 9, 29,
     30, 40, 32, 41, 147, 22, 29, 46, 23, 4, 31, 39, 15, 32, 27, 32, 34, 17,
     46, 42, 43, 34, 19]

X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')
w = tf.Variable(0.0, name='weights')
b = tf.Variable(0.0, name='bias')

Y_predicted = X * w + b

loss = tf.square(Y - Y_predicted, name='loss')

optimizer = tf.train.GradientDescentOptimizer(
    learning_rate=.001).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    # writer = tf.summary.FileWriter(logs_dir, sess.graph)
    sess.run(init)
    for i in range(100):
        total_loss = 0
        for j in range(len(x)):
            _, current_loss = sess.run(
                [optimizer, loss], feed_dict={X: x[j], Y: y[j]})
            total_loss += current_loss
        print("Epoch {} finished. Average Loss: {}".format(
            i, total_loss / len(x)))
    w_value, b_value = sess.run([w, b])
    print("Final weights: {}".format(w_value))
    print("Final bias: {}".format(b_value))
# writer.close


best_fit_x = [0, 40]
best_fit_y = [b_value, 40 * w_value + b_value]
plt.plot(x, y, 'r.', best_fit_x, best_fit_y)
