import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from rdkit import Chem
from argparse import ArgumentParser
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

parser = ArgumentParser()
parser.add_argument('--csv_file', type=str, default='data/cmtd202100108-sup-0001-mlr.data',
                    help='path to file containing the data of tantillo')
parser.add_argument('--csv-file-steroids', type=str, default='cmtd202100108-sup-0001-bdefr.data',
                    help='path to file containing the data of tantillo')
parser.add_argument('--csv-file-steroids', type=str, default='stereoids.csv',
                    help='path to file containing the data of tantillo')
parser.add_argument('--pkl-file', type=str, default='clean_data.pkl',
                    help='path to file containing the data of tantillo')
parser.add_argument('--pkl-file_pred', type=str, default='clean_data_pred.pkl',
                    help='path to file containing the data of tantillo with predictions')
parser.add_argument('--pkl-file_pred', type=str, default='clean_data_steroids_pred.pkl',
                    help='path to file containing the data of tantillo steroids with predictions')


def convert_to_csv():

    data = pd.read_csv('data/cmtd202100108-sup-0001-mlr.data', delim_whitespace=True)
    data.to_csv('data/tantillo.csv')
    data = pd.read_csv('data/cmtd202100108-sup-0001-bdefr.data', delim_whitespace=True)
    data.to_csv('data/tantillo_steroids.csv')


def clean_data():
    """ Clean and modify the data from Tantillo paper """

    data = pd.read_csv('data/tantillo_smiles.csv', delimiter=';', index_col=0)
    data_steroids = pd.read_csv('data/tantillo_steroids_smiles.csv', delimiter=';', index_col=0)

    pair_mol_rad = data['SMILES'].tolist()

    smi_pairs = []
    smi_maps = []

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        mol, rad = map_rxn(mol, rad)
        smi_pairs.append(f"{mol}.{rad}")
        smi_maps.append(mol)
        smi_maps.append(rad)

    data['rxn_smile'] = smi_pairs
    pair_mol_rad = data_steroids['SMILES'].tolist()

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        smi_maps.append(canonicalize_smiles(mol))
        smi_maps.append(canonicalize_smiles(rad))

    smi_maps = list(set(smi_maps))
    smiles = pd.DataFrame(smi_maps, columns=['smiles'])
    smiles.to_csv('data/predict_data_tantillo.csv')

    data.to_csv('data/clean_data_tantillo.csv')
    data.to_pickle('data/clean_data_tantillo.pkl')

    return None


def add_pred():

    data = pd.read_pickle('data/clean_data_tantillo.pkl')
    pred = pd.read_pickle('data/pred_reactivity_tantillo.pkl')

    spin_rad = []
    q_rad = []
    q_mol = []
    q_molH = []
    bdfe = []
    fr_bde = []
    bv = []

    for row in data.itertuples():
        mol_smiles, rad_smiles = row.rxn_smile.split('.')
        bdfe.append(pred.loc[pred['smiles'] == rad_smiles].dG.values[0])
        fr_bde.append(pred.loc[pred['smiles'] == rad_smiles].frozen_dG.values[0])
        bv.append(pred.loc[pred['smiles'] == rad_smiles].Buried_Vol.values[0])
        idx_rad = get_rad_index(rad_smiles)
        idx_mol, idx_molh = get_mol_index(rad_smiles, mol_smiles, idx_rad)
        spin_rad.append(pred.loc[pred['smiles'] == rad_smiles].spin_densities.values[0][idx_rad])
        q_mol.append(pred.loc[pred['smiles'] == mol_smiles].charges_all_atom.values[0][idx_mol])
        q_molH.append(pred.loc[pred['smiles'] == mol_smiles].charges_all_atom.values[0][idx_molh])
        q_rad.append(pred.loc[pred['smiles'] == rad_smiles].charges_all_atom.values[0][idx_rad])

    data['s_rad'] = spin_rad
    data['q_rad'] = q_rad
    data['q_mol'] = q_mol
    data['q_molH'] = q_molH
    data['Buried_Vol'] = bv
    data['BDFE'] = bdfe
    data['fr_BDE'] = fr_bde

    data.to_pickle('data/input_tantillo.pkl')
    data.to_csv('data/input_tantillo.csv')

    return None


def remove_points():

    data = pd.read_pickle('data/input_tantillo.pkl')
    data_corr = data.corr(numeric_only=True)
    data_new = data.drop([2, 7,  14, 16, 20, 21])
    data_new_corr = data_new.corr(numeric_only=True)
    data_new.reset_index(inplace=True)
    data_new.to_pickle('data/input_tantillo_wo.pkl')
    data_new.to_csv('data/input_tantillo_wo.csv')


def canonicalize_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def get_rad_index(smiles):
    """ Get the index of the radical atom"""

    mol = Chem.MolFromSmiles(smiles)

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            idx = atom.GetIdx()

    return idx


def get_mol_index(rad_smiles, mol_smiles, rad_idx):
    """ Get the index of the radical atom in the molecule and of the H"""

    os_mol = Chem.MolFromSmiles(rad_smiles)
    cs_mol = Chem.MolFromSmiles(mol_smiles)

    substructure = os_mol.GetSubstructMatch(cs_mol)

    if not substructure:
        Chem.Kekulize(cs_mol, clearAromaticFlags=True)
        Chem.Kekulize(os_mol, clearAromaticFlags=True)
        substructure = os_mol.GetSubstructMatch(cs_mol)

    mol_idx = substructure.index(rad_idx)

    cs_mol = Chem.AddHs(cs_mol)

    atom = [atom for atom in cs_mol.GetAtoms() if atom.GetIdx() == mol_idx][0]

    h_idx = [ngb.GetIdx() for ngb in atom.GetNeighbors() if ngb.GetSymbol() == 'H'][0]

    return mol_idx, h_idx


def map_rxn(smi_mol, smi_rad):

    mol = Chem.MolFromSmiles(smi_mol)
    rad = Chem.MolFromSmiles(smi_rad)

    substruct = rad.GetSubstructMatch(mol)

    # Now we will have some problems because some radicals are not aromatic, so the command won't find any substructmatch

    sanitize = False

    if not substruct:
        Chem.Kekulize(mol, clearAromaticFlags=True)
        Chem.Kekulize(rad, clearAromaticFlags=True)
        substruct = rad.GetSubstructMatch(mol)
        sanitize = True

    if not substruct:
        return False

    for idx, a_mol in enumerate(mol.GetAtoms()):
        a_mol.SetAtomMapNum(idx + 1)

    for idx_mol, idx_rad in enumerate(substruct):
        atom_rad = rad.GetAtomWithIdx(idx_rad)
        atom_mol = mol.GetAtomWithIdx(idx_mol)
        map_number = atom_mol.GetAtomMapNum()
        atom_rad.SetAtomMapNum(map_number)

    # and now we have to "re-aromatize" the aromatic molecules or radicals"

    if sanitize:
        Chem.SanitizeMol(mol)
        Chem.SanitizeMol(rad)

    smi_map_mol = Chem.MolToSmiles(mol)
    smi_map_rad = Chem.MolToSmiles(rad)

    return smi_map_mol, smi_map_rad


if __name__ == '__main__':
    # set up
    args = parser.parse_args()