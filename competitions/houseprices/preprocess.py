import pandas as pd
import tensorflow as tf
import numpy as np
import os
import math
from inspect import getsourcefile

# get the root directory so we can import our models without
# needing it in our python path
current_path = os.path.abspath(getsourcefile(lambda: 0))
current_dir = os.path.dirname(current_path)


LEARNING_RATE = 0.001
DATA_DIR = current_dir + '/data/'
CHECKPOINT_DIR = current_dir + '/checkpoint/'
PREDICTION_DIR = current_dir + \
    '/prediction/linear-model-lr-{}-'.format(LEARNING_RATE)

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


def preprocess_data(data_df, train_df):
    y = data_df['SalePrice'].values.reshape(
        (-1, 1)) if 'SalePrice' in data_df else None
    continuous = data_df[CONTINUOUS_VARIABLES]

    # standardize based on the training data mean and stdev, not the test data
    standardized = (continuous - train_df.mean()) / train_df.std()

    # create a random standard normal data frame of the same shape
    dfrand = pd.DataFrame(
        data=np.random.standard_normal(size=standardized.shape),
        columns=standardized.columns,
        index=standardized.index
    )

    # use the random standard normal values to fill in the NaNs
    standardized = standardized.fillna(dfrand)

    features = standardized.as_matrix()

    # convert categorical variables to one hot tensors
    with tf.Session() as sess:
        for col, unique in CATEGORICAL_VARIABLES.items():
            values = data_df[col].values
            depth = len(unique)
            unique_dict = {unique[u]: u for u in range(depth)}
            new_vals = []
            for v in values:
                key = 'nan' if type(v) is float and math.isnan(v) else v
                new_vals.append(unique_dict[key])

            one_hot = sess.run(tf.one_hot(new_vals, depth, dtype=tf.int16))
            features = np.concatenate((features, one_hot), axis=1)

    return features, y


def get_data(train_file, test_file=None):
    data_df = pd.read_csv(DATA_DIR + train_file)
    split = int(len(data_df) * 0.8)
    train_df, valid_df = data_df[:split], data_df[split:]

    train_features, train_y = preprocess_data(train_df, train_df)

    if test_file:
        test_df = pd.read_csv(DATA_DIR + test_file)
        test_features, test_y = preprocess_data(test_df, train_df)
        return test_features, test_y, None, None
    else:
        valid_features, valid_y = preprocess_data(valid_df, train_df)
        return train_features, train_y, valid_features, valid_y
