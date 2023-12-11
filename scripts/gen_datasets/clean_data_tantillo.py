import pandas as pd
from rdkit import Chem


def convert_to_csv():

    data = pd.read_csv('../../data/data_tantillo/cmtd202100108-sup-0001-mlr.data', delim_whitespace=True)
    data.to_csv('data/tantillo.csv')
    data = pd.read_csv('../../data/data_tantillo/cmtd202100108-sup-0001-bdefr.data', delim_whitespace=True)
    data.to_csv('data/tantillo_steroids.csv')


def clean_data():
    """ Clean and modify the data from Tantillo paper """

    data = pd.read_csv('../../data/data_tantillo/tantillo_smiles.csv', delimiter=';', index_col=0)
    data_steroids = pd.read_csv('../../data/data_tantillo/tantillo_steroids_smiles.csv', delimiter=';', index_col=0)

    pair_mol_rad = data['SMILES'].tolist()

    smi_pairs = []
    all_smiles = []

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        mol = canonicalize_smiles(mol)
        rad = canonicalize_smiles(rad)
        smi_pairs.append(f"{mol}.{rad}")
        all_smiles.append(mol)
        all_smiles.append(rad)

    data['rxn_smile'] = smi_pairs
    pair_mol_rad = data_steroids['SMILES'].tolist()
    smi_pairs = []

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        mol = canonicalize_smiles(mol)
        rad = canonicalize_smiles(rad)
        smi_pairs.append(f"{mol}.{rad}")
        all_smiles.append(mol)
        all_smiles.append(rad)

    all_smiles = list(set(all_smiles))

    data_steroids['rxn_smile'] = smi_pairs
    smiles = pd.DataFrame(all_smiles, columns=['smiles'])
    smiles.to_csv('../../data/data_tantillo/species_reactivity_tantillo_dataset.csv')

    data.to_csv('../../data/data_tantillo/clean_data_tantillo.csv')
    data.to_pickle('../../data/data_tantillo/clean_data_tantillo.pkl')

    data_steroids.to_csv('../../data/data_tantillo/clean_data_steroids_tantillo.csv')
    data_steroids.to_pickle('../../data/data_tantillo/clean_data_steroids_tantillo.pkl')

    return None


def canonicalize_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


if __name__ == '__main__':
    clean_data()
