import tensorflow as tf
import os
import math
import csv
import sys
from preprocess import get_data
from inspect import getsourcefile

# get the root directory so we can import our models without
# needing it in our python path
current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
current_dir_split = current_dir.rsplit(os.path.sep)
current_dir_split = current_dir_split[:len(current_dir_split) - 2]
root_dir = os.path.sep.join(current_dir_split)

sys.path.append(root_dir)

# now we can import our linear model
from models.LinearRegression import LinearRegressionModel  # noqa: E402


LEARNING_RATE = 0.001
DATA_DIR = current_dir + '/data/'
CHECKPOINT_DIR = current_dir + '/checkpoint/'
PREDICTION_DIR = current_dir + \
    '/prediction/linear-model-lr-{}-'.format(LEARNING_RATE)


def predict():
    features, _, _, _ = get_data('train.csv', 'test.csv')

    num_samples, num_features = features.shape
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    model = LinearRegressionModel(X, Y, LEARNING_RATE, num_features)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        prediction = sess.run(model.prediction, feed_dict={X: features})

        with open(PREDICTION_DIR + 'best-model-predictions.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Id', 'SalePrice'])
            for i in range(len(prediction)):
                writer.writerow([i + 1461, prediction[i][0]])


def train():
    train_features, train_y, valid_features, valid_y = get_data('train.csv')
    # features, y = preprocess_data('train.csv')

    num_samples, num_features = train_features.shape
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    model = LinearRegressionModel(X, Y, LEARNING_RATE, num_features)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        # ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)

        best_validation_loss = math.inf
        improvement_threshold = 1.0
        validation_frequency = 5000
        num_epochs = 10000
        epoch_increase = 5000
        epoch = 0
        while (epoch < num_epochs):
            epoch = epoch + 1
            _, current_loss, global_step = sess.run(
                [model.optimize, model.error, model.global_step],
                feed_dict={X: train_features, Y: train_y})

            new_validation_loss = sess.run(model.error, feed_dict={
                                           X: valid_features, Y: valid_y})

            # check if validation is improving every few training epochs
            if epoch % validation_frequency == 0:
                if new_validation_loss < best_validation_loss:
                    # if it is better by more than the threshold then keep
                    # training
                    if new_validation_loss < best_validation_loss * improvement_threshold \
                            and epoch > num_epochs - epoch_increase:
                        num_epochs = num_epochs + epoch_increase

                    best_validation_loss = new_validation_loss

                    # save the best model
                    saver.save(sess, CHECKPOINT_DIR +
                               'linear-model-lr-{}-best-model'.format(
                                   LEARNING_RATE),
                               global_step=model.global_step)

            print("Epoch {} finished. Average Loss: {}, Validation Error: {}, Best Validation Error: {}"
                  .format(global_step, current_loss, new_validation_loss, best_validation_loss))


if __name__ == '__main__':
    train()
    # predict()
