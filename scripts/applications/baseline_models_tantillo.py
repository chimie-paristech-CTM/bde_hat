import pandas as pd
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import logging

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default='data/input_tantillo_wo.pkl',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--steroids_file', type=str, default='data/input_steroids_tantillo.pkl',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--features', nargs="+", type=str, default=['s_rad', 'Buried_Vol', 'BDFE', 'q_rad'],
                    help='features for the different models')
# interactive way
parser.add_argument("--mode", default='client', action="store", type=str)
parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
parser.add_argument("--port", default=57546, action="store", type=int)


def final_eval(df_train, df_test, model, target_column='DFT_Barrier'):
    """
    Function to perform cross-validation

    Args:
        df_train (pd.DataFrame): the DataFrame containing features and targets for train
        df_test (pd.DataFrame): the DataFrame containing features and targets for test
        model (sklearn.Regressor): An initialized sklearn model
        target_column (str): target column

    Returns:
        int: the obtained RMSE and MAE
    """

    feature_names = [column for column in df_train.columns if column not in['DFT_Barrier', 'rxn_id']]

    df_train = df_train.sample(frac=1, random_state=0)

    X_train, y_train = df_train[feature_names], df_train[[target_column]]
    X_test, y_test = df_test[feature_names], df_test[[target_column]]
    # scale the two dataframes
    feature_scaler = StandardScaler()
    feature_scaler.fit(X_train)
    X_train = feature_scaler.transform(X_train)
    X_test = feature_scaler.transform(X_test)

    target_scaler = StandardScaler()
    target_scaler.fit(y_train)
    y_train = target_scaler.transform(y_train)
    y_test = target_scaler.transform(y_test)

    # fit and compute rmse and mae
    model.fit(X_train, y_train.ravel())
    predictions = model.predict(X_test)
    predictions = predictions.reshape(-1,1)

    rmse = np.sqrt(mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
    mae = mean_absolute_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test))
    r2 = r2_score(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(predictions))

    return rmse, mae, r2


def get_accuracy_linear_regression(df_train, df_test, logger):

    model = LinearRegression()
    rmse, mae, r2 = final_eval(df_train, df_test, model)

    logger.info(f'RMSE, MAE and R^2 for linear regression: {rmse} {mae} {r2}')


def create_logger(name='output.log') -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :name: The directory in which to save the logs.
    :return: The logger.
    """

    logger = logging.getLogger('final.log')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create file handler which logs even debug messages
    fh = logging.FileHandler(name)
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def prepare_df(data, features):

    features = features + ['DFT_Barrier'] + ['rxn_id']

    columns_remove = [column for column in data.columns if column not in features]

    df = data.drop(columns=columns_remove)

    return df


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    df = pd.read_pickle(args.input_file)
    df_exp = pd.read_pickle(args.steroids_file)
    logger = create_logger(name='output_tantillo.log')
    features = args.features
    df = prepare_df(df, features)
    df_exp = prepare_df(df_exp, features)
    logger.info(f'The considered features are: {features}')

    # Linear regression
    get_accuracy_linear_regression(df, df_exp, logger)
