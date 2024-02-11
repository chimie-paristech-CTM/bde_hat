import pandas as pd
from argparse import ArgumentParser
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from hyperopt import fmin, tpe
from functools import partial
import logging
from types import SimpleNamespace
from hyperopt import hp

n_estimator_dict = {0: 10, 1: 30, 2: 50, 3: 100, 4: 150, 5: 200, 6: 300, 7: 400, 8: 600}
min_samples_leaf_dict = {0: 1, 1: 2, 2: 5, 3: 10, 4: 20, 5: 50}

parser = ArgumentParser()
parser.add_argument('--input_file', type=str, default='data/input_omega_ffnn.pkl',
                    help='path to file containing the train set')
parser.add_argument('--exp_file', type=str, default='data/input_omega_exp_ffnn.pkl',
                    help='path to file containing the exp set')
parser.add_argument('--selectivity_file', type=str, default='data/input_omega_selectivity_ffnn.pkl',
                    help='path to file containing the exp set')
parser.add_argument('--features', nargs="+", type=str, default=['dG_forward', 'dG_reverse', 'q_reac0',
                                                                'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0',
                                                                's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1',
                                                                'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse'],
                    help='features for the different models')
# interactive way
parser.add_argument("--mode", default='client', action="store", type=str)
parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
parser.add_argument("--port", default=57546, action="store", type=int)


def final_eval(df_train, df_test, model, target_column='G_act'):
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

    feature_names = [column for column in df_train.columns if column not in['rxn_id', 'DG_TS', 'G_r', 'DG_TS_tunn', 'G_act']]

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


def cross_val(df, model, n_folds, target_column='G_act'):
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
    rmse_list, mae_list, r2_list= [], [], []
    feature_names = [column for column in df.columns if column not in['rxn_id', 'DG_TS', 'G_r', 'DG_TS_tunn', 'gibbs_exp']]

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

        r2_fold = r2_score(target_scaler.inverse_transform(y_test), target_scaler.inverse_transform(predictions))
        r2_list.append(r2_fold)


    rmse = np.mean(np.array(rmse_list))
    mae = np.mean(np.array(mae_list))
    r2 = np.mean(np.array(r2_list))

    return rmse, mae, r2


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


def get_optimal_parameters_lasso(df, logger):
    """
    Get the optimal descriptors for xgboost (descriptors) through Bayesian optimization.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object

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
    rmse, mae, r2, ev = final_eval(df_train, df_test, model)

    logger.info(f'RMSE, MAE, R^2 and explained variance for linear regression: {rmse} {mae} {r2} {ev}')


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


def get_accuracy_rf_descriptors(df_train, df_test, logger, parameters):
    """
    Get the random forest (descriptors) accuracy in cross-validation.

    Args:
        df (pd.DataFrame): input dataframe
        logger (logging.Logger): logger-object
        n_fold (int): number of folds to use during cross-validation
        parameters (Dict): a dictionary containing the parameters to be used
        split_dir (str, optional): path to the directory containing the splits. Defaults to None.
    """
    model = RandomForestRegressor(n_estimators=int(parameters['n_estimators']),
            max_features=parameters['max_features'], min_samples_leaf=int(parameters['min_samples_leaf']))
    rmse, mae, r2 = final_eval(df_train, df_test, model)
    logger.info(f'RMSE, MAE, R^2 and for RF: {rmse} {mae} {r2}')


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
    cval, _, _ = cross_val(data, estimator, 4)


    return cval.mean()


def exp_data_corr_linear(df_train, features):

    df_exp_target = pd.read_csv('data/cumoexp.csv', index_col=0)
    df_exp_features = pd.read_pickle('data/input_omega_exp_ffnn.pkl')

    # train a model, predict the DG_TS in bietti's dataset
    model = LinearRegression()
    X_train, y_train = df_train[features], df_train[['DG_TS']]
    X_test = df_exp_features[features]
    model.fit(X_train, y_train)
    model.coef_  # array([[0.51643321]]) = a
    model.intercept_ #  array([9.00419397]) = b
    y_pred = model.predict(X_test)

    # scaled the experimental values
    model = LinearRegression()
    X_train, y_train = df_exp[['gibbs_exp']], y_pred
    model.fit(X_train, y_train)
    model.coef_  # array([[0.88245116]]) = a
    model.intercept_  # array([5.36774428]) = b
    y_exp_corr = model.predict(X_train)
    mae = mean_absolute_error(y_train, y_exp_corr)  # 1.64
    rmse = np.sqrt(mean_squared_error(y_train, y_exp_corr)) # 1.92
    r2 = r2_score(y_train, y_exp_corr) # 0.31


def empirical_model():
    """
        Empirical model omega paper
            dG_act = alpha*dG_rxn*(1-d) + beta*dX^2 + delta
    """

    alpha = 0.41
    beta = - 0.20
    delta = 19.70

    df_train = pd.read_csv('data/train_original_omega.csv', index_col=0)
    df_train['dG_rxn'] = df_train['BDFE_sub'] - df_train['BDFE_cat']
    df_train['dX'] = df_train['x_sub'] - df_train['x_cat']
    df_train['d_houk'] = df_train['d'].apply(lambda x: 0.44 if x else 0)

    df_train['dG_act_pred'] = alpha * df_train['dG_rxn'] * (1 - df_train['d_houk']) + beta * (df_train['dX']**2) + delta

    mae_train = mean_absolute_error(df_train['gibbs'], df_train['dG_act_pred'])
    r2_train = r2_score(df_train['gibbs'], df_train['dG_act_pred'])
    rmse_train = np.sqrt(mean_squared_error(df_train['gibbs'], df_train['dG_act_pred']))

    df_test = pd.read_csv('data/test_original_omega.csv', index_col=0)
    df_test['dG_rxn'] = df_test['BDFE_sub'] - df_test['BDFE_cat']
    df_test['dX'] = df_test['x_sub'] - df_test['x_cat']
    df_test['d_houk'] = df_test['d'].apply(lambda x: 0.44 if x else 0)

    df_test['dG_act_pred'] = alpha * df_test['dG_rxn'] * (1 - df_test['d_houk']) + beta * (df_test['dX']**2) + delta

    mae_test = mean_absolute_error(df_test['gibbs'], df_test['dG_act_pred'])
    r2_test = r2_score(df_test['gibbs'], df_test['dG_act_pred'])
    rmse_test = np.sqrt(mean_squared_error(df_test['gibbs'], df_test['dG_act_pred']))



def prepare_df(data, features):

    features = features + ['rxn_id'] + ['DG_TS'] + ['G_act']

    columns_remove = [column for column in data.columns if column not in features]

    df = data.drop(columns=columns_remove)

    return df


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    df = pd.read_pickle(args.input_file)
    df_train = df.iloc[:240]
    df_train = df_train.loc[df_train['G_act'] != 'FALSE']
    df_test = df.iloc[240:]
    df_exp = pd.read_pickle(args.exp_file)
    df_selectivity = pd.read_pickle(args.selectivity_file)
    logger = create_logger()
    features = args.features
    df = prepare_df(df, features)
    df_train = prepare_df(df_train, features)
    df_test = prepare_df(df_test, features)
    logger.info(f'The considered features are: {features}')

    # Linear regression
    get_accuracy_linear_regression(df_train, df_train, logger)
    get_accuracy_linear_regression(df_train, df_test, logger)

    # RF
    optimal_parameters_rf = get_optimal_parameters_rf_descriptors(df_train, logger, 32)
    get_accuracy_rf_descriptors(df_train, df_test, logger, optimal_parameters_rf)

    # exp
    #df_exp = pd.read_pickle('data/input_omega_exp_ffnn_1.pkl')
    #features += ['gibbs_exp']
    #df_exp = prepare_df(df_exp, features)
    #rmse, mae, r2, ev = cross_val(df_exp, LinearRegression(), 10, target_column='gibbs_exp')
    #logger.info(f'Experimental data of bietti')
    #logger.info(f'RMSE, MAE, R^2 and explained variance for 10 folds linear regression: {rmse} {mae} {r2} {ev}')
    #rmse, mae, r2, ev = cross_val(df_exp, RandomForestRegressor(), 10, target_column='gibbs_exp')
    #logger.info(f'RMSE, MAE, R^2 and explained variance for 10 folds RF: {rmse} {mae} {r2} {ev}')

    #exp corr