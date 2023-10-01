#!/usr/bin/python
import pandas as pd
from argparse import ArgumentParser
from lib.utils import create_logger
from lib.final_functions import prepare_df
from lib.final_functions import get_optimal_parameters_knn_fp
from lib.final_functions import get_cross_val_accuracy_linear_regression, get_cross_val_accuracy_knn_fp
from lib.final_functions import get_optimal_parameters_rf_descriptors, get_optimal_parameters_rf_fp
from lib.final_functions import get_cross_val_accuracy_rf_descriptors, get_cross_val_accuracy_rf_fps
from lib.final_functions import get_optimal_parameters_xgboost_descriptors, get_optimal_parameters_xgboost_fp
from lib.final_functions import get_cross_val_accuracy_xgboost_descriptors, get_cross_val_accuracy_xgboost_fps
from lib.fingerprints import get_fingerprints_all_rxn


parser = ArgumentParser()
parser.add_argument('--csv-file', type=str, default='../../data/reactivity_database_mapped.csv',
                    help='path to file containing the rxn-smiles')
parser.add_argument('--input-file', type=str, default='../../data/input_ffnn.pkl',
                    help='path to the input file')
parser.add_argument('--split_dir', type=str, default='data_files/splits_cross_validation',
                    help='path to the folder containing the requested splits for the cross validation')
parser.add_argument('--n-fold', type=int, default=10,
                    help='the number of folds to use during cross validation')
parser.add_argument('--features', nargs="+", type=str, default=['dG_forward', 'dG_reverse', 'q_reac0',
                                                                'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0',
                                                                's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1',
                                                                'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse'],
                    help='features for the different models')
# interactive way
parser.add_argument("--mode", default='client', action="store", type=str)
parser.add_argument("--host", default='127.0.0.1', action="store", type=str)
parser.add_argument("--port", default=57546, action="store", type=int)


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    logger = create_logger()
    df = prepare_df(args.input_file, args.features)
    #df_rxn_smiles = pd.read_csv(args.csv_file, index_col=0)
    #df_fps = get_fingerprints_all_rxn(df_rxn_smiles)
    n_fold = args.n_fold
    split_dir = args.split_dir
    features = args.features
    logger.info(f'The considered features are: {features}')

    # linear regression
    get_cross_val_accuracy_linear_regression(df, logger, n_fold)

    # KNN fingerprints
    #optimal_parameters_knn = get_optimal_parameters_knn_fp(df_fps, logger, max_eval=64)
    #get_cross_val_accuracy_knn_fp(df_fps, logger, n_fold, optimal_parameters_knn)

    # RF descriptors
    optimal_parameters_rf_descs = get_optimal_parameters_rf_descriptors(df, logger, max_eval=64)
    get_cross_val_accuracy_rf_descriptors(df, logger, n_fold, optimal_parameters_rf_descs)

    # RF fingerprints
    #optimal_parameters_rf_fps = get_optimal_parameters_rf_fp(df_fps, logger, max_eval=64)
    #get_cross_val_accuracy_rf_fps(df_fps, logger, n_fold, optimal_parameters_rf_fps)

    # XGboost descriptors
    optimal_parameters_xgboost_descs = get_optimal_parameters_xgboost_descriptors(df, logger, max_eval=128)
    get_cross_val_accuracy_xgboost_descriptors(df, logger, n_fold, optimal_parameters_xgboost_descs)

    # XGboost fingerprints
    #optimal_parameters_xgboost_fp = get_optimal_parameters_xgboost_fp(df_fps, logger, max_eval=128)
    #get_cross_val_accuracy_xgboost_fps(df_fps, logger, n_fold, optimal_parameters_xgboost_fp)

