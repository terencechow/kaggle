import pandas as pd
import tensorflow as tf
import numpy as np
import os
import math
import csv
import sys
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

DATA_DIR = current_dir + '/data/'
CHECKPOINT_DIR = current_dir + '/checkpoint/'
PREDICTION_DIR = current_dir + '/prediction/'

CONTINUOUS_VARIABLES = [
    'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'MoSold', 'YrSold'
]

CATEGORICAL_VARIABLES = {
    'MSSubClass': [160, 70, 40, 75, 45, 80, 50, 20, 85, 180, 30, 120, 90, 60, 190, 150],
    'MSZoning': ['RL', 'FV', 'C (all)', 'RH', 'RM', 'nan'],
    'Street': ['Pave', 'Grvl'],
    'Alley': ['nan', 'Pave', 'Grvl'],
    'LotShape': ['IR1', 'Reg', 'IR2', 'IR3'],
    'LandContour': ['Lvl', 'Bnk', 'Low', 'HLS'],
    'Utilities': ['NoSeWa', 'AllPub', 'nan'],
    'LotConfig': ['FR3', 'Corner', 'Inside', 'FR2', 'CulDSac'],
    'LandSlope': ['Sev', 'Gtl', 'Mod'],
    'Neighborhood': ['Gilbert', 'SWISU', 'SawyerW', 'MeadowV', 'OldTown', 'NoRidge', 'Edwards', 'Timber', 'Veenker', 'Blueste', 'Somerst', 'Crawfor', 'NAmes', 'StoneBr', 'BrkSide', 'NWAmes', 'Mitchel', 'NridgHt', 'BrDale', 'IDOTRR', 'Sawyer', 'NPkVill', 'Blmngtn', 'CollgCr', 'ClearCr'],
    'Condition1': ['RRAe', 'Norm', 'RRNn', 'RRNe', 'PosA', 'PosN', 'Feedr', 'RRAn', 'Artery'],
    'Condition2': ['RRAe', 'Norm', 'RRNn', 'PosA', 'PosN', 'Feedr', 'RRAn', 'Artery'],
    'BldgType': ['2fmCon', 'Twnhs', 'TwnhsE', 'Duplex', '1Fam'],
    'HouseStyle': ['1.5Unf', '1.5Fin', 'SFoyer', 'SLvl', '2.5Fin', '2Story', '2.5Unf', '1Story'],
    'OverallQual': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'OverallCond': [1, 2, 3, 4, 5, 6, 7, 8, 9],
    'RoofStyle': ['Gable', 'Flat', 'Hip', 'Shed', 'Gambrel', 'Mansard'],
    'RoofMatl': ['CompShg', 'Roll', 'WdShngl', 'Membran', 'ClyTile', 'Metal', 'Tar&Grv', 'WdShake'],
    'Exterior1st': ['VinylSd', 'BrkComm', 'ImStucc', 'Wd Sdng', 'BrkFace', 'MetalSd', 'CemntBd', 'AsbShng', 'Plywood', 'CBlock', 'Stucco', 'AsphShn', 'WdShing', 'Stone', 'HdBoard', 'nan'],
    'Exterior2nd': ['VinylSd', 'ImStucc', 'CmentBd', 'Brk Cmn', 'Wd Sdng', 'Other', 'BrkFace', 'MetalSd', 'CBlock', 'AsbShng', 'Plywood', 'Stucco', 'AsphShn', 'Stone', 'Wd Shng', 'HdBoard', 'nan'],
    'MasVnrType': ['nan', 'BrkCmn', 'None', 'BrkFace', 'Stone'],
    'ExterQual': ['Ex', 'TA', 'Gd', 'Fa'],
    'ExterCond': ['Ex', 'Fa', 'TA', 'Po', 'Gd'],
    'Foundation': ['Slab', 'PConc', 'BrkTil', 'CBlock', 'Wood', 'Stone'],
    'BsmtQual': ['nan', 'Ex', 'Fa', 'TA', 'Gd'],
    'BsmtCond': ['nan', 'Fa', 'TA', 'Po', 'Gd'],
    'BsmtExposure': ['nan', 'Mn', 'Av', 'Gd', 'No'],
    'BsmtFinType1': ['nan', 'GLQ', 'Rec', 'LwQ', 'BLQ', 'Unf', 'ALQ'],
    'BsmtFinType2': ['nan', 'GLQ', 'Rec', 'LwQ', 'BLQ', 'Unf', 'ALQ'],
    'Heating': ['Floor', 'Grav', 'Wall', 'GasW', 'GasA', 'OthW'],
    'HeatingQC': ['Ex', 'Fa', 'TA', 'Po', 'Gd'],
    'CentralAir': ['N', 'Y'],
    'Electrical': ['nan', 'FuseA', 'FuseP', 'SBrkr', 'FuseF', 'Mix'],
    'KitchenQual': ['Ex', 'TA', 'Gd', 'Fa', 'nan'],
    'Functional': ['Sev', 'Maj1', 'Maj2', 'Min1', 'Min2', 'Typ', 'Mod', 'nan'],
    'FireplaceQu': ['nan', 'Ex', 'Fa', 'TA', 'Po', 'Gd'],
    'GarageType': ['nan', 'Basment', 'BuiltIn', 'CarPort', '2Types', 'Detchd', 'Attchd'],
    'GarageFinish': ['Unf', 'RFn', 'Fin', 'nan'],
    'GarageQual': ['nan', 'Ex', 'Fa', 'TA', 'Po', 'Gd'],
    'GarageCond': ['nan', 'Ex', 'Fa', 'TA', 'Po', 'Gd'],
    'PavedDrive': ['P', 'N', 'Y'],
    'PoolQC': ['nan', 'Ex', 'Gd', 'Fa'],
    'Fence': ['nan', 'MnPrv', 'GdPrv', 'MnWw', 'GdWo'],
    'MiscFeature': ['nan', 'Gar2', 'TenC', 'Othr', 'Shed'],
    'SaleType': ['ConLw', 'Oth', 'WD', 'Con', 'New', 'CWD', 'COD', 'ConLD', 'ConLI', 'nan'],
    'SaleCondition': ['Family', 'Partial', 'AdjLand', 'Abnorml', 'Normal', 'Alloca']
}


def preprocess_data(filename):
    df = pd.read_csv(DATA_DIR + filename)
    y = df['SalePrice'].values.reshape((-1, 1)) if 'SalePrice' in df else None
    features = df[CONTINUOUS_VARIABLES]

    # replace NANs with average for continuous variables
    features = features.fillna(features.mean())
    features = features.as_matrix()

    # convert categorical variables to one hot tensors
    with tf.Session() as sess:
        for col, unique in CATEGORICAL_VARIABLES.items():
            values = df[col].values
            depth = len(unique)
            unique_dict = {unique[u]: u for u in range(depth)}
            new_vals = []
            for v in values:
                key = 'nan' if type(v) is float and math.isnan(v) else v
                new_vals.append(unique_dict[key])

            one_hot = sess.run(tf.one_hot(new_vals, depth, dtype=tf.int16))
            features = np.concatenate((features, one_hot), axis=1)

    return features, y


def predict():
    features, _ = preprocess_data('test.csv')

    num_samples, num_features = features.shape
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    model = LinearRegressionModel(X, Y, 0.00000000001, num_features)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # features, _ = preprocess_data('test.csv')

    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        prediction = sess.run(model.prediction, feed_dict={X: features})

        with open(PREDICTION_DIR + 'predictions.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Id', 'SalePrice'])
            for i in range(len(prediction)):
                writer.writerow([i + 1461, prediction[i][0]])


def train():
    features, y = preprocess_data('train.csv')

    num_samples, num_features = features.shape
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    model = LinearRegressionModel(X, Y, 0.00000000001, num_features)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(100000):
            _, current_loss, global_step = sess.run(
                [model.optimize, model.error, model.global_step],
                feed_dict={X: features, Y: y})

            print("Epoch {} finished. Average Loss: {}".format(
                global_step, current_loss))

            if i > 0 and global_step % 1000 == 0:
                saver.save(sess, CHECKPOINT_DIR, global_step=model.global_step)


if __name__ == '__main__':
    train()
    # predict()
