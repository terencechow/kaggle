import pandas as pd
import tensorflow as tf
import numpy as np
from LinearRegressionModel import LinearRegressionModel
import os
import math
import csv

DATA_DIR = 'data/houseprices/'
MODEL_DIR = 'models/houseprices/'

CONTINUOUS_VARIABLES = [
    'LotFrontage', 'LotArea', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',
    'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath',
    'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',
    'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',
    'MiscVal', 'MoSold', 'YrSold'
]

# CATEGORICAL_VARIABLES = [
#     'MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour',
#     'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
#     'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond',
#     'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
#     'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond',
#     'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC',
#     'CentralAir', 'Electrical', 'KitchenQual', 'Functional', 'FireplaceQu',
#     'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive',
#     'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition'
# ]
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
    y = df['SalePrice'].values if 'SalePrice' in df else None
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
    features, _ = preprocess_data('train.csv')

    num_samples, num_features = features.shape
    X = tf.placeholder(tf.float32, name='X')
    Y = tf.placeholder(tf.float32, name='Y')

    model = LinearRegressionModel(X, Y, 0.0001, num_features)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    features, _ = preprocess_data('test.csv')

    with tf.Session() as sess:
        sess.run(init)
        ckpt = tf.train.get_checkpoint_state(os.path.dirname(MODEL_DIR))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        prediction = sess.run(model.prediction, feed_dict={X: features})
        with open('predictions.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['Id', 'SalePrice'])
            for i in range(len(prediction)):
                writer.writerow([i + 1461, prediction[i]])


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

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(MODEL_DIR))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        for i in range(50000):
            total_loss = 0
            for j in range(num_samples):
                sess.run(model.optimize, feed_dict={X: features[j], Y: y[j]})

                current_loss, global_step = sess.run(
                    [model.error, model.global_step], feed_dict={X: features, Y: y})

                total_loss += current_loss

            print("Epoch {} finished. Average Loss: {}".format(
                global_step / num_samples, total_loss / num_samples))

            if i > 0 and (global_step / num_samples) % 100 == 0:
                saver.save(sess, MODEL_DIR, global_step=model.global_step)


if __name__ == '__main__':
    train()
