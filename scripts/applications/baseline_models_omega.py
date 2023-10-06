import pandas as pd
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from hyperopt import hp
from xgboost import XGBRegressor
from types import SimpleNamespace
from hyperopt import fmin, tpe
from functools import partial
import logging

n_estimator_dict = {0: 10, 1: 30, 2: 50, 3: 100, 4: 150, 5: 200, 6: 300, 7: 400, 8: 600}
min_samples_leaf_dict = {0: 1, 1: 2, 2: 5, 3: 10, 4: 20, 5: 50}

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default='data/input_omega_ffnn.pkl',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--features', nargs="+", type=str, default=['dG_forward', 'dG_reverse', 'q_reac0',
                                                                'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0',
                                                                's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1',
                                                                'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse'],
                    help='features for the different models')
# interactive way
parser.add_argument("--mode", default='client', action="store", type=str)
parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
parser.add_argument("--port", default=57546, action="store", type=int)


def final_eval(df_train, df_test, model, target_column='DG_TS'):
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

    feature_names = [column for column in df_train.columns if column not in['rxn_id', 'DG_TS', 'G_r', 'DG_TS_tunn']]

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

    return rmse, mae


def cross_val(df, model, n_folds, target_column='DG_TS'):
    """
    Function to perform cross-validation

    Args:
        df (pd.DataFrame): the DataFrame containing features and targets
        model (sklearn.Regressor): An initialized sklearn model
        n_folds (int): the number of folds
        target_column (str): target column

    Returns:
        int: the obtained RMSE and MAE
    """
    rmse_list, mae_list = [], []
    feature_names = [column for column in df.columns if column not in['rxn_id', 'DG_TS', 'G_r', 'DG_TS_tunn']]

    df = df.sample(frac=1, random_state=0)
    chunk_list = np.array_split(df, n_folds)

    for i in range(n_folds):

        df_train = pd.concat([chunk_list[j] for j in range(n_folds) if j != i])
        df_test = chunk_list[i]

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

        rmse_fold = np.sqrt(mean_squared_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test)))
        rmse_list.append(rmse_fold)

        mae_fold = mean_absolute_error(target_scaler.inverse_transform(predictions), target_scaler.inverse_transform(y_test))
        mae_list.append(mae_fold)

    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))

    return rmse, mae


def bayesian_opt(df, space, objective, model_class, n_train=0.8, max_eval=32):
    """
    Overarching function for Bayesian optimization

    Args:
        df (pd.DataFrame): dataframe containing the data points
        space (dict): dictionary containing the parameters for the selected regressor
        objective (function): specific objective function to be used
        model_class (Model): the abstract model class to initialize in every iteration
        n_train (float, optional): fraction of the training data to use. Defaults to 0.8.
        max_eval (int, optional): number of iterations to perform. Defaults to 32

    Returns:
        dict: optimal parameters for the selected regressor
    """
    df_sample = df.sample(frac=n_train)
    fmin_objective = partial(objective, data=df_sample, model_class=model_class)
    best = fmin(fmin_objective, space, algo=tpe.suggest, max_evals=max_eval)

    return best


def get_optimal_parameters_rf_descriptors(df, logger, max_eval=32):
    """
    Get the optimal descriptors for random forest (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'n_estimators': hp.choice('n_estimators', [10, 30, 50, 100, 150, 200, 300, 400, 600]),
        'max_features': hp.quniform('max_features', low=0.1, high=1, q=0.1),
        'min_samples_leaf': hp.choice('min_samples_leaf', [1, 2, 5, 10, 20, 50])
    }

    optimal_parameters = bayesian_opt(df, space, objective_rf, RandomForestRegressor, max_eval=max_eval)
    optimal_parameters['n_estimators'] = n_estimator_dict[optimal_parameters['n_estimators']]
    optimal_parameters['min_samples_leaf'] = min_samples_leaf_dict[optimal_parameters['min_samples_leaf']]
    logger.info(f'Optimal parameters for RF -- descriptors: {optimal_parameters}')

    return optimal_parameters


def objective_rf(args_dict, data, model_class):
    """
    Objective function for random forest Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the RF regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(n_estimators=int(args.n_estimators),
                                max_features=args.max_features,
                                min_samples_leaf=int(args.min_samples_leaf),
                                random_state=2)
    cval, _ = cross_val(data, estimator, 4)

    return cval.mean()


def get_accuracy_rf_descriptors(df_train, df_test, logger, parameters):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df_train (pd.DataFrame): train dataframe
        df_test (pd.DataFrame): test dataframe
        logger (logging.Logger): logger-object
        parameters (Dict): a dictionary containing the parameters to be used
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']),
            max_features=parameters['max_features'], min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae = final_eval(df_train, df_test, model)
    logger.info(f'RMSE and MAE for RF -- descriptors: {rmse} {mae}')


def get_optimal_parameters_xgboost_descriptors(df, logger, max_eval=32):
    """
    Get the optimal descriptors for xgboost (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """
    space = {
        'max_depth': hp.quniform('max_depth', low=2, high=10, q=1),
        'gamma': hp.qloguniform('gamma', low=0.0, high=6.0, q=2.0),
        'n_estimators': hp.quniform('n_estimators', low=100, high=800, q=100),
        'learning_rate': hp.quniform('learning_rate', low=0.05, high=0.20, q=0.05),
        'min_child_weight': hp.quniform('min_child_weight', low=2, high=10, q=2.0)
    }
    optimal_parameters = bayesian_opt(df, space, objective_xgboost, XGBRegressor, max_eval=max_eval)
    logger.info(f'Optimal parameters for xgboost -- descriptors: {optimal_parameters}')

    return optimal_parameters


def objective_xgboost(args_dict, data, model_class):
    """
    Objective function for xgboost Bayesian optimization

    Args:
        args_dict (dict): dictionary containing the parameters for the xgboost regressor
        data (pd.DataFrame): dataframe containing the data points
        model_class (Model): the abstract model class to initialize in every iteration

    Returns:
        float: the cross-validation score (RMSE)
    """
    args = SimpleNamespace(**args_dict)
    estimator = model_class(max_depth=int(args.max_depth),
                                    gamma=args.gamma,
                                    learning_rate=args.learning_rate,
                                    min_child_weight=args.min_child_weight,
                                    n_estimators=int(args.n_estimators))
    cval,_ = cross_val(data, estimator, 4)

    return cval.mean()


def get_accuracy_xgboost_descriptors(df_train, df_test, logger, parameters):
    """
    Get the xgboost (descriptors) accuracy in cross-validation.

    Args:
        df_train (pd.DataFrame): train dataframe
        df_test (pd.DataFrame): test dataframe
        logger (logging.Logger): logger-object
        parameters (Dict): a dictionary containing the parameters to be used
    """
    model = XGBRegressor(max_depth=int(parameters['max_depth']),
                        gamma=parameters['gamma'],
                        n_estimators=int(parameters['n_estimators']),
                        learning_rate=parameters['learning_rate'],
                        min_child_weight=parameters['min_child_weight'])
    rmse, mae = final_eval(df_train, df_test, model)
    logger.info(f'RMSE and MAE for xgboost -- descriptors: {rmse} {mae}')


def get_optimal_parameters_lasso(df, logger):
    """
    Get the optimal descriptors for xgboost (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        max_eval (int, optional): number of BO evaluations

    returns:
        Dict: a dictionary containing the optimal parameters
    """

    space = np.arange(0.02, 1, 0.02)
    df = df.sample(frac=0.8)
    cval = [cross_val(df, Lasso(alpha=alpha), 4)[0] for alpha in space]
    optimal_parameters = {'alpha': min(cval)}
    logger.info(f'Optimal parameters for lasso -- descriptors: {optimal_parameters}')

    return optimal_parameters


def get_accuracy_lasso(df_train, df_test, logger, parameters):
    """
    Get the xgboost (descriptors) accuracy in cross-validation.

    Args:
        df_train (pd.DataFrame): train dataframe
        df_test (pd.DataFrame): test dataframe
        logger (logging.Logger): logger-object
        parameters (Dict): a dictionary containing the parameters to be used
    """

    model = Lasso(alpha=(parameters['alpha']))
    rmse, mae = final_eval(df_train, df_test, model)
    logger.info(f'RMSE and MAE for LASSO -- descriptors: {rmse} {mae}')


def get_accuracy_linear_regression(df_train, df_test, logger):

    model = LinearRegression()
    rmse, mae = final_eval(df_train, df_test, model)

    logger.info(f'RMSE and MAE for linear regression: {rmse} {mae}')

def create_logger() -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param save_dir: The directory in which to save the logs.
    :return: The logger.
    """

    logger = logging.getLogger('final.log')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create file handler which logs even debug messages
    fh = logging.FileHandler('output.log')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def prepare_df(data, features):

    features = features + ['rxn_id'] + ['DG_TS']

    columns_remove = [column for column in data.columns if column not in features]

    df = data.drop(columns=columns_remove)

    return df


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    df = pd.read_pickle(args.input_file)
    df_train = df.iloc[:240]
    df_test = df.iloc[240:]
    df_test.reset_index(inplace=True)
    df_test.drop(columns=['index'], inplace=True)
    logger = create_logger()
    features = args.features
    df_train = prepare_df(df_train, features)
    df_test = prepare_df(df_test, features)
    logger.info(f'The input file is: {args.input_file}')
    logger.info(f'The considered features are: {features}')

    # Linear regression
    get_accuracy_linear_regression(df_train, df_test, logger)

    # LASSO
    optimal_parameters_lasso = get_optimal_parameters_lasso(df_train, logger)
    get_accuracy_lasso(df_train, df_test, logger, optimal_parameters_lasso)

    # RF descriptors
    optimal_parameters_rf_descs = get_optimal_parameters_rf_descriptors(df_train, logger, max_eval=64)
    get_accuracy_rf_descriptors(df_train, df_test, logger, optimal_parameters_rf_descs)

    # XGboost descriptors
    optimal_parameters_xgboost_descs = get_optimal_parameters_xgboost_descriptors(df_train, logger, max_eval=128)
    get_accuracy_xgboost_descriptors(df_train, df_test, logger, optimal_parameters_xgboost_descs)
