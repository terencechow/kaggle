import tensorflow as tf
import os
# import math
import csv
import sys
from preprocess import get_data
from inspect import getsourcefile
import numpy as np

# get the root directory so we can import our models without
# needing it in our python path
current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)
current_dir_split = current_dir.rsplit(os.path.sep)
current_dir_split = current_dir_split[:len(current_dir_split) - 2]
root_dir = os.path.sep.join(current_dir_split)

sys.path.append(root_dir + '/models')

# now we can import our model
from MultilayerPerception import MultilayerPerceptionModel  # noqa: E402


LEARNING_RATE = 0.0000001
DATA_DIR = current_dir + '/data/'
CHECKPOINT_DIR = current_dir + '/checkpoint/'
PREDICTION_DIR = current_dir + \
    '/prediction/mlp-lr-{:.0e}-'.format(LEARNING_RATE)
LOG_DIR = current_dir + '/logs/'


def predict():
    features, _, _, _ = get_data('train.csv', 'test.csv')

    num_samples, num_features = features.shape
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    hidden_layers = [
        {"name": 'h1', "input_size": num_features,
         "output_size": 1000, "activation": tf.nn.relu},
        {"name": 'h2', "output_size": 1, "activation": lambda x: x}
    ]

    model = MultilayerPerceptionModel(X, Y, hidden_layers, LEARNING_RATE)

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


X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')


def train(skip_validation=False):
    train_features, train_y, valid_features, valid_y = get_data('train.csv')
    # features, y = preprocess_data('train.csv')

    num_samples, num_features = train_features.shape
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')
    hidden_layers = [
        {"name": 'h1', "input_size": num_features,
         "output_size": 500, "activation": tf.nn.relu},
        {"name": 'h2', "output_size": 1, "activation": lambda x: x}
    ]

    model = MultilayerPerceptionModel(X, Y, hidden_layers, LEARNING_RATE, 0.5)

    merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        train_writer = tf.summary.FileWriter(
            LOG_DIR + 'mlp-lr-{:.0e}'.format(LEARNING_RATE), sess.graph)

        # ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        # if ckpt and ckpt.model_checkpoint_path:
        #     saver.restore(sess, ckpt.model_checkpoint_path)

        best_validation_loss = float("inf")  # math.inf for python3
        improvement_threshold = 1.0
        validation_frequency = 5000
        num_epochs = 100000
        epoch_increase = 5000
        epoch = 0

        while (epoch < num_epochs):
            epoch = epoch + 1
            _, current_loss, global_step, summary = sess.run(
                [model.optimize, model.error, model.global_step, merged],
                feed_dict={X: train_features, Y: train_y})

            new_validation_loss = sess.run(model.error, feed_dict={
                                           X: valid_features, Y: valid_y})
            # check if validation is improving every few training epochs
            if not skip_validation and epoch % validation_frequency == 0:
                if new_validation_loss < best_validation_loss:
                    # if it is better by more than the threshold then keep
                    # training
                    if new_validation_loss < best_validation_loss * improvement_threshold \
                            and epoch > num_epochs - epoch_increase:
                        num_epochs = num_epochs + epoch_increase

                    best_validation_loss = new_validation_loss

                    # save the best model
                    saver.save(sess, CHECKPOINT_DIR +
                               'mlp-lr-{:.0e}-valid-error-{:.4e}'.format(
                                   LEARNING_RATE, best_validation_loss),
                               global_step=model.global_step)

            # print("Epoch {} finished. Average Loss: {:.4e}, Validation Error: {:.4e}, Best Validation Error: {:.4e}"
            #       .format(global_step, current_loss, new_validation_loss, best_validation_loss))

            print 'Epoch {} finished. Average Loss: {:.4e}, Validation Error: {:.4e}, Best Validation Error: {:.4e}'.format(global_step, current_loss, new_validation_loss, best_validation_loss)

            if (np.isnan(current_loss)):
                break
            train_writer.add_summary(summary, epoch)

    train_writer.close()


if __name__ == '__main__':
    train(skip_validation=True)
    # predict()
