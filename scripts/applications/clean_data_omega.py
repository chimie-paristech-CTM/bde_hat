import pandas as pd
import numpy as np
from rdkit import Chem
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns

parser = ArgumentParser()
parser.add_argument('--csv-file', type=str, default='data/data_omega.csv',
                    help='path to file containing the data of ACS Omega')
parser.add_argument('--csv-file_mapped', type=str, default='clean_data_mapped.csv',
                    help='path to file containing the clean mapped data of ACS Omega')
parser.add_argument('--test_set', type=str, default='test_set_omega_mapped.csv',
                    help='path to file containing the test set data of ACS Omega')
parser.add_argument('--pred_GNN', type=str, default='test_predicted_acs_omega.csv',
                    help='path to file containing the output from the GNN')
parser.add_argument('--train_set_baseline', type=str, default='train_input_baseline_acs_omega_bv_fr.pkl',
                    help='path to file containing the train set for the baseline models')
parser.add_argument('--test_set_baseline', type=str, default='test_input_baseline_acs_omega_bv_fr.pkl',
                    help='path to file containing the test set for the baseline models')
parser.add_argument('--dataset', type=str, default='input_acs_omega.pkl', help='path to the data'),
parser.add_argument('--features', nargs="+", type=str, default=['dG_forward', 'dG_reverse', 'q_reac0',
                                                                'qH_reac0', 'q_reac1', 's_reac1', 'q_prod0',
                                                                's_prod0', 'q_prod1', 'qH_prod1', 'BV_reac1', 'BV_prod0',
                                                                'fr_dG_forward', 'fr_dG_reverse'],
                    help='features for the different models')


def clean_data(csv_file):
    """ Clean and modify the data from ACS Omega paper """

    data = pd.read_csv(csv_file, delimiter=';')
    data.rename(columns={'Col 1': 'Substrate_Idx', 'Col 2': 'Mol.Rad', 'Col 3':  'BDFE', 'Col 4': 'G_rxn',
                 'Col 5': 'Electronegativity', 'Col 6': 'delta_Electronegativity_squared',
                 'Col 7': 'Buried_Volume', 'Col 8': 'G_act'}, inplace=True)

    pair_mol_rad = data['Mol.Rad'].tolist()

    smi_rad = []
    smi_mol = []

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        smi_rad.append(canonicalize_smiles(rad))
        smi_mol.append(canonicalize_smiles(mol))

    methoxyl_rad = canonicalize_smiles('C[O]')
    trifluoroethoxyl_rad = canonicalize_smiles('C(F)(F)(F)C[O]')
    cumo_rad = canonicalize_smiles('C1=CC=CC=C1C([O])(C)C')
    tbutoxil_rad = canonicalize_smiles('C(C)(C)(C)[O]')
    trichloroethoxyl_rad = canonicalize_smiles('C(Cl)(Cl)(Cl)C[O]')
    methoxyl_mol = canonicalize_smiles('CO')
    trifluoroethoxyl_mol = canonicalize_smiles('C(F)(F)(F)CO')
    cumo_mol = canonicalize_smiles('C1=CC=CC=C1C(O)(C)C')
    tbutoxil_mol = canonicalize_smiles('C(C)(C)(C)O')
    trichloroethoxyl_mol = canonicalize_smiles('C(Cl)(Cl)(Cl)CO')

    set_rad = [(methoxyl_rad, methoxyl_mol), (trifluoroethoxyl_rad, trifluoroethoxyl_mol), (cumo_rad, cumo_mol),
               (tbutoxil_rad, tbutoxil_mol), (trichloroethoxyl_rad, trichloroethoxyl_mol)]

    rxns = []

    j = 0
    count = 0
    for mol, rad in zip(smi_mol, smi_rad):
        i = set_rad[j]
        rxns.append(f"{mol}.{i[0]}>>{rad}.{i[1]}")
        count += 1
        if count == 60:
            j += 1
            count = 0

    idx = [i for i in range(300)]

    data['ID'] = idx
    data['RXN_SMILES'] = rxns
    data['mol'] = smi_mol
    data['rad'] = smi_rad
    data.drop(columns=['Mol.Rad'], inplace=True)

    data_autodE_format = pd.DataFrame()
    data_autodE_format['rxn_id'] = idx
    data_autodE_format['RXN_SMILES'] = rxns
    data_autodE_format['dG_rxn'] = data['G_rxn']
    data_autodE_format['dG_act'] = data['G_act']
    data_autodE_format['dG_act'] = data['G_act']

    data_autodE_format.to_csv('data/reactivity_omega_database.csv')

    data.to_csv('data/clean_data_omega.csv')
    data.to_pickle('data/clean_data_omega.pkl')

    return None


def canonicalize_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def corr_descriptors(X_train, name):

    # compute the correlation matrix
    corr = np.abs(X_train.corr())

    # generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool_)
    mask[np.triu_indices_from(mask)] = True

    # set up the matplotlib figure
    f, ax = plt.subplots(figsize=(16, 14))

    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, vmin=0.0, vmax=1.0, center=0.5, linewidths=.1, cmap="rocket_r", cbar_kws={"shrink": .8})

    plt.tight_layout()
    plt.show()
    plt.savefig(f'{name}.png')


def reacs_prods(smiles_rxn : str) -> str:
    """ Return the components of the rxn """

    dot_index = smiles_rxn.index('.')
    sdot_index = smiles_rxn.index('.', dot_index + 1)
    limit_index = smiles_rxn.index('>>')
    reac1 = smiles_rxn[:dot_index]
    reac2 = smiles_rxn[dot_index + 1: limit_index]
    prod1 = smiles_rxn[limit_index + 2: sdot_index]
    prod2 = smiles_rxn[sdot_index + 1:]

    return reac1, reac2, prod1, prod2


def split_data():

    df = pd.read_pickle('data/input_omega_ffnn.pkl')
    df_train = df.iloc[:240]
    df_test = df.iloc[240:]
    df_train.to_pickle('data/train_omega_ffnn.pkl')
    df_test.to_pickle('data/test_omega_ffnn.pkl')
    df_train.to_csv('data/train_omega_ffnn.csv')
    df_test.to_csv('data/test_omega_ffnn.csv')

if __name__ == '__main__':
    # set up
    args = parser.parse_args()

