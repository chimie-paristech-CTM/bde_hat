#!/usr/bin/python
import pandas as pd
from argparse import ArgumentParser
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import logging

parser = ArgumentParser()
parser.add_argument('--pred', type=str, default='../../data/test_preds.pickle',
                    help='path to file containing the predictions of surrogate model')
parser.add_argument('--smiles_file', type=str, default='../../data/test_smiles.pickle',
                    help='path to file containing the smiles')
parser.add_argument('--all_data', type=str, default='../../data/dataset_surrogate_model_cs.pkl',
                    help='path to file containing the complete data')
parser.add_argument('--test_set', type=str, default='../../data/testset_surrogate_model.pkl',
                    help='path to file containing the test set, in case you have an specific test set')
parser.add_argument('--logger', type=str, default='surrogate',
                    help='name for the logger')


def checking_pred(pred_file, smiles_file, dataset_file, logger):

    pred = pd.read_pickle(pred_file)

    smiles_test = pd.read_pickle(smiles_file)

    dataset = pd.read_pickle(dataset_file)

    test_set = pd.DataFrame()

    test_set['smiles'] = smiles_test
    test_set['dG_pred'] = pred[3]

    test_atom_descriptor = pd.DataFrame()
    test_atom_descriptor['spin_densities_pred'] = pd.DataFrame(pred[0])
    test_atom_descriptor['charges_all_atom_pred'] = pd.DataFrame(pred[1])

    matching_elements = dataset.loc[dataset['smiles'].isin(test_set['smiles'])]
    merged_data = pd.merge(test_set, matching_elements, on='smiles', how='inner')

    test_atom_descriptor['spin_densities'] = merged_data.spin_densities.explode('spin_densities')
    test_atom_descriptor['charges_all_atom'] = merged_data.charges_all_atom.explode('charges_all_atom')

    merged_data['dG'] = merged_data['dG'].apply(lambda x: float(x[0]))
    test_atom_descriptor['spin_densities'] = test_atom_descriptor['spin_densities'].apply(lambda x: float(x))
    test_atom_descriptor['charges_all_atom'] = test_atom_descriptor['charges_all_atom'].apply(lambda x: float(x))

    # Calculate RMSE
    rmse_dG = np.sqrt(mean_squared_error(merged_data['dG'], merged_data['dG_pred']))
    rmse_q_all_atoms = np.sqrt(
        mean_squared_error(test_atom_descriptor['charges_all_atom'], test_atom_descriptor['charges_all_atom_pred']))
    rmse_spin_all_atoms = np.sqrt(
        mean_squared_error(test_atom_descriptor['spin_densities'], test_atom_descriptor['spin_densities_pred']))

    # Calculate MAE
    mae_dG = mean_absolute_error(merged_data['dG'], merged_data['dG_pred'])
    mae_q_all_atoms = mean_absolute_error(test_atom_descriptor['charges_all_atom'],
                                          test_atom_descriptor['charges_all_atom_pred'])
    mae_spin_all_atoms = mean_absolute_error(test_atom_descriptor['spin_densities'],
                                             test_atom_descriptor['spin_densities_pred'])

    # Calculate R^2
    r2_dG = r2_score(merged_data['dG'], merged_data['dG_pred'])
    r2_q_all_atom = r2_score(test_atom_descriptor['charges_all_atom'], test_atom_descriptor['charges_all_atom_pred'])
    r2_spin_all_atoms = r2_score(test_atom_descriptor['spin_densities'], test_atom_descriptor['spin_densities_pred'])

    logger.info(f"dG:                     RMSE {rmse_dG:.4f} MAE {mae_dG:.4f} R^2 {r2_dG:.4f}")
    logger.info(f"charge all atoms:       RMSE {rmse_q_all_atoms:.4f} MAE {mae_q_all_atoms:.4f} R^2 {r2_q_all_atom:.4f}")
    logger.info(f"spin densities:         RMSE {rmse_spin_all_atoms:.4f} MAE {mae_spin_all_atoms:.4f} R^2 {r2_spin_all_atoms:.4f}")

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

    sns.kdeplot(data=merged_data, x='dG', y='dG_pred', ax=axes[0], cmap='rocket_r', fill=True, thresh=0)
    sns.kdeplot(data=test_atom_descriptor, x='charges_all_atom', y='charges_all_atom_pred', ax=axes[1], cmap='rocket_r',
                fill=True, thresh=0)
    sns.kdeplot(data=test_atom_descriptor, x='spin_densities', y='spin_densities_pred', ax=axes[2], cmap='rocket_r',
                fill=True, thresh=0)

    axes[0].set_title("dG")
    axes[1].set_title("charges")
    axes[2].set_title("spin_densities")

    # Hide tick labels on the x-axis and y-axis
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    axes[2].set_xlabel('')
    axes[2].set_ylabel('')

    # Add MAE and RMSE annotations to each subplot
    axes[0].text(0.02, 0.8, f"MAE: {mae_dG:.3f}\nRMSE: {rmse_dG:.3f}\n$R^2$: {r2_dG:.3f}",
                 transform=axes[0].transAxes, fontsize=12)
    axes[1].text(0.02, 0.8, f"MAE:  {mae_q_all_atoms:.3f}\nRMSE: {rmse_q_all_atoms:.3f}\n$R^2$: {r2_q_all_atom:.3f}",
                 transform=axes[1].transAxes, fontsize=12)
    axes[2].text(0.02, 0.8,
                 f"MAE:  {mae_spin_all_atoms:.3f}\nRMSE: {rmse_spin_all_atoms:.3f}\n$R^2$: {r2_spin_all_atoms:.3f}",
                 transform=axes[2].transAxes, fontsize=12)

    fig.text(0.5, 0.0015, "Computed Value", ha='center')
    fig.text(0.0015, 0.5, "Predicted Value", va='center', rotation='vertical')

    plt.tight_layout()

    plt.savefig('correlation_plot_kde_full_data.png')


def checking_pred_split(pred_file, smiles_file, testset_file, logger):
    """ How in the data there are closed shell molecules and the dG = 0, we should calculate the stats without it"""

    pred = pd.read_pickle(pred_file)

    smiles_test = pd.read_pickle(smiles_file)

    dataset = pd.read_pickle(testset_file)
    dataset.reset_index(inplace=True)

    test_set = pd.DataFrame()

    test_set['smiles'] = smiles_test
    test_set['Buried_Vol_pred'] = pred[2]
    test_set['dG_pred'] = pred[3]
    test_set['frozen_dG_pred'] = pred[4]

    test_atom_descriptor = pd.DataFrame()
    test_atom_descriptor['spin_densities_pred'] = pd.DataFrame(pred[0])
    test_atom_descriptor['charges_all_atom_pred'] = pd.DataFrame(pred[1])

    test_atom_descriptor['spin_densities'] = dataset.spin_densities.explode('spin_densities')
    test_atom_descriptor['charges_all_atom'] = dataset.charges_all_atom.explode('charges_all_atom')

    test_set['dG'] = dataset['dG'].apply(lambda x: float(x[0]))
    test_set['frozen_dG'] = dataset['frozen_dG'].apply(lambda x: float(x[0]))
    test_set['Buried_Vol'] = dataset['Buried_Vol'].apply(lambda x: float(x[0]))

    data_wo_cs = test_set.loc[test_set.dG != 0.]

    test_atom_descriptor['spin_densities'] = test_atom_descriptor['spin_densities'].apply(lambda x: float(x))
    test_atom_descriptor['charges_all_atom'] = test_atom_descriptor['charges_all_atom'].apply(lambda x: float(x))
    test_atom_desc_wo_cs = test_atom_descriptor.loc[test_atom_descriptor.spin_densities != 0.]

    # Calculate RMSE
    rmse_dG = np.sqrt(mean_squared_error(data_wo_cs['dG'], data_wo_cs['dG_pred']))
    rmse_fr_dG = np.sqrt(mean_squared_error(data_wo_cs['frozen_dG'], data_wo_cs['frozen_dG_pred']))
    rmse_BV = np.sqrt(mean_squared_error(data_wo_cs['Buried_Vol'], data_wo_cs['Buried_Vol_pred']))
    rmse_q_all_atoms = np.sqrt(
        mean_squared_error(test_atom_descriptor['charges_all_atom'], test_atom_descriptor['charges_all_atom_pred']))
    rmse_spin_all_atoms = np.sqrt(
        mean_squared_error(test_atom_desc_wo_cs['spin_densities'], test_atom_desc_wo_cs['spin_densities_pred']))

    # Calculate MAE
    mae_dG = mean_absolute_error(data_wo_cs['dG'], data_wo_cs['dG_pred'])
    mae_fr_dG = mean_absolute_error(data_wo_cs['frozen_dG'], data_wo_cs['frozen_dG_pred'])
    mae_BV = mean_absolute_error(data_wo_cs['Buried_Vol'], data_wo_cs['Buried_Vol_pred'])
    mae_q_all_atoms = mean_absolute_error(test_atom_descriptor['charges_all_atom'],
                                          test_atom_descriptor['charges_all_atom_pred'])
    mae_spin_all_atoms = mean_absolute_error(test_atom_desc_wo_cs['spin_densities'],
                                             test_atom_desc_wo_cs['spin_densities_pred'])

    # Calculate R^2
    r2_dG = r2_score(data_wo_cs['dG'], data_wo_cs['dG_pred'])
    r2_fr_dG = r2_score(data_wo_cs['frozen_dG'], data_wo_cs['frozen_dG_pred'])
    r2_BV = r2_score(data_wo_cs['Buried_Vol'], data_wo_cs['Buried_Vol_pred'])
    r2_q_all_atom = r2_score(test_atom_descriptor['charges_all_atom'], test_atom_descriptor['charges_all_atom_pred'])
    r2_spin_all_atoms = r2_score(test_atom_desc_wo_cs['spin_densities'], test_atom_desc_wo_cs['spin_densities_pred'])

    logger.info(f"dG:                     RMSE {rmse_dG:.4f} MAE {mae_dG:.4f} R^2 {r2_dG:.4f}")
    logger.info(f"frozen dG:              RMSE {rmse_fr_dG:.4f} MAE {mae_fr_dG:.4f} R^2 {r2_fr_dG:.4f}")
    logger.info(f"Buried Volume:          RMSE {rmse_BV:.4f} MAE {mae_BV:.4f} R^2 {r2_BV:.4f}")
    logger.info(f"partial charges:        RMSE {rmse_q_all_atoms:.4f} MAE {mae_q_all_atoms:.4f} R^2 {r2_q_all_atom:.4f}")
    logger.info(f"spin densities:         RMSE {rmse_spin_all_atoms:.4f} MAE {mae_spin_all_atoms:.4f} R^2 {r2_spin_all_atoms:.4f}")

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(16, 8))

    sns.kdeplot(data=data_wo_cs, x='dG', y='dG_pred', ax=axes[0, 0], cmap='rocket_r', fill=True)
    sns.kdeplot(data=data_wo_cs, x='Buried_Vol', y='Buried_Vol_pred', ax=axes[0, 1], cmap='rocket_r', fill=True)
    sns.kdeplot(data=data_wo_cs, x='frozen_dG', y='frozen_dG_pred', ax=axes[0, 2], cmap='rocket_r', fill=True)
    sns.kdeplot(data=test_atom_descriptor, x='charges_all_atom', y='charges_all_atom_pred', ax=axes[1, 0], fill=True,
                cmap='rocket_r')
    sns.kdeplot(data=test_atom_desc_wo_cs, x='spin_densities', y='spin_densities_pred', ax=axes[1, 1], cmap='rocket_r',
                fill=True)

    axes[0, 0].set_title("BDFE (kcal/mol)")
    axes[0, 1].set_title("buried volume")
    axes[0, 2].set_title("frozen BDFE (kcal/mol)")
    axes[1, 0].set_title("partial charges (e)")
    axes[1, 1].set_title("spin densities (e)")

    # Hide tick labels on the x-axis and y-axis
    axes[0, 0].set_xlabel('')
    axes[0, 0].set_ylabel('')
    axes[0, 1].set_xlabel('')
    axes[0, 1].set_ylabel('')
    axes[0, 2].set_xlabel('')
    axes[0, 2].set_ylabel('')
    axes[1, 0].set_xlabel('')
    axes[1, 0].set_ylabel('')
    axes[1, 1].set_xlabel('')
    axes[1, 1].set_ylabel('')

    # Add MAE and RMSE annotations to each subplot
    axes[0, 0].text(0.02, 0.8, f"MAE: {mae_dG:.3f}\nRMSE: {rmse_dG:.3f}\n$R^2$: {r2_dG:.3f}",
                 transform=axes[0, 0].transAxes, fontsize=10)
    axes[0, 1].text(0.02, 0.8, f"MAE: {mae_BV:.3f}\nRMSE: {rmse_BV:.3f}\n$R^2$: {r2_BV:.3f}",
                    transform=axes[0, 1].transAxes, fontsize=10)
    axes[0, 2].text(0.02, 0.8, f"MAE: {mae_fr_dG:.3f}\nRMSE: {rmse_fr_dG:.3f}\n$R^2$: {r2_fr_dG:.3f}",
                    transform=axes[0, 2].transAxes, fontsize=10)
    axes[1, 0].text(0.02, 0.8, f"MAE:  {mae_q_all_atoms:.3f}\nRMSE: {rmse_q_all_atoms:.3f}\n$R^2$: {r2_q_all_atom:.3f}",
                 transform=axes[1, 0].transAxes, fontsize=10)
    axes[1, 1].text(0.02, 0.8,
                 f"MAE:  {mae_spin_all_atoms:.3f}\nRMSE: {rmse_spin_all_atoms:.3f}\n$R^2$: {r2_spin_all_atoms:.3f}",
                 transform=axes[1, 1].transAxes, fontsize=10)

    fig.text(0.5, 0.01, "Computed Value", ha='center', fontsize=12)
    fig.text(0.005, 0.5, "Predicted Value", va='center', rotation='vertical', fontsize=12)

    plt.delaxes(axes[1, 2])
    plt.tight_layout(pad=1.5)

    plt.savefig('correlation_plot_kde_full_data_without_values_zero_4.png')



def checking_pred_split_bv_fr():
    """ TEST FUNCTION FOR EVALUATED THE SURROGATE MODEL OF BV AND FROZEN BDFE"""

    pred_file = 'results/bv_fr_opt/test_preds.pickle'
    smiles_file = 'results/bv_fr_opt/test_smiles.pickle'
    dataset_file = '../../data/surrogate_data_bv_fr.pkl'

    pred = pd.read_pickle(pred_file)

    smiles_test = pd.read_pickle(smiles_file)

    dataset = pd.read_pickle(dataset_file)

    test_set = pd.DataFrame()

    test_set['smiles'] = smiles_test
    test_set['Buried_Vol_pred'] = pred[0]
    test_set['BDFE_fr_pred'] = pred[1]

    matching_elements = dataset.loc[dataset['smiles'].isin(test_set['smiles'])]
    merged_data = pd.merge(test_set, matching_elements, on='smiles', how='inner')

    merged_data['Buried_Vol'] = merged_data['Buried_Vol'].apply(lambda x: float(x[0]))
    merged_data['BDFE_fr'] = merged_data['BDFE_fr'].apply(lambda x: float(x[0]))

    data_wo_cs = merged_data.loc[merged_data.BDFE_fr != 0.]

    # Calculate RMSE
    rmse_bv = np.sqrt(mean_squared_error(data_wo_cs['Buried_Vol'], data_wo_cs['Buried_Vol_pred']))
    rmse_bdfe_fr = np.sqrt(mean_squared_error(data_wo_cs['BDFE_fr'], data_wo_cs['BDFE_fr_pred']))

    # Calculate MAE
    mae_bv = mean_absolute_error(data_wo_cs['Buried_Vol'], data_wo_cs['Buried_Vol_pred'])
    mae_bdfe_fr = mean_absolute_error(data_wo_cs['BDFE_fr'], data_wo_cs['BDFE_fr_pred'])


    # Calculate R^2
    r2_bv = r2_score(data_wo_cs['Buried_Vol'], data_wo_cs['Buried_Vol_pred'])
    r2_bdfe_fr = r2_score(data_wo_cs['BDFE_fr'], data_wo_cs['BDFE_fr_pred'])

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.kdeplot(data=data_wo_cs, x='Buried_Vol', y='Buried_Vol_pred', ax=axes[0], cmap='rocket_r', fill=True)
    sns.kdeplot(data=data_wo_cs, x='BDFE_fr', y='BDFE_fr_pred', ax=axes[1], cmap='rocket_r', fill=True)

    axes[0].set_title("Buried Volume")
    axes[1].set_title("BDFE$_{frozen}$  (kcal/mol)")

    # Hide tick labels on the x-axis and y-axis
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')

    # Add MAE and RMSE annotations to each subplot
    axes[0].text(0.02, 0.780, f"MAE: {mae_bv:.3f}\nRMSE: {rmse_bv:.3f}\n$R^2$: {r2_bv:.3f}",
                 transform=axes[0].transAxes, fontsize=14)
    axes[1].text(0.02, 0.780, f"MAE:  {mae_bdfe_fr:.3f}\nRMSE: {rmse_bdfe_fr:.3f}\n$R^2$: {r2_bdfe_fr:.3f}",
                 transform=axes[1].transAxes, fontsize=14)

    fig.text(0.5, 0.01, "Computed Value", ha='center', fontsize=14)
    fig.text(0.005, 0.5, "Predicted Value", va='center', rotation='vertical', fontsize=14)

    plt.tight_layout(pad=1.5)

    plt.savefig('correlation_plot_kde_full_data_without_values_zero_bv_fr.png')


def create_plot_1M(b):
    """ df is the huge dataset of 1M"""

    mean = b.dG.mean()
    std = b.dG.std()
    sns.histplot(b, x='dG', kde=True, color='darkred')
    plt.xlabel('"$\Delta$G$_{rxn}$ (kcal/mol)"')
    plt.title('')

    plt.tight_layout()
    plt.show()

    plt.savefig('histogram plots 1M.png')


def create_logger(name: str) -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: The directory in which to save the logs.
    :return: The logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create file handler which logs even debug messages
    fh = logging.FileHandler('output_{}.log'.format(name))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


if __name__ == '__main__':
    # set up
    args = parser.parse_args()
    logger = create_logger(args.logger)
    checking_pred_split(args.pred, args.test_smiles, args.all_data, logger)