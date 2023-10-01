import pandas as pd
from argparse import ArgumentParser
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

parser = ArgumentParser()
parser.add_argument('--csv-file', type=str, default='../../data/reactivity_database_mapped.csv',
                    help='path to file containing the reactivity dataset')
parser.add_argument('--pred_surrogate', type=str, default='../../data/species_reactivity_dataset_pred.pkl',
                    help='path to file containing the predictions')


def gnn_desc(csv_file, pred_surrogate):

    df_mapped = pd.read_csv(csv_file, index_col=0)
    df_pred = pd.read_pickle(pred_surrogate)

    spin_dens = []
    dGs = []
    BVs = []
    fr_dGs = []

    for row in df_pred.itertuples():

        if is_radical(row.smiles):

            spin_dens.append(row.spin_densities)
            dGs.append(row.dG)
            BVs.append(row.Buried_Vol)
            fr_dGs.append(row.frozen_dG)

        else:

            spin_densities_mol = np.zeros(num_atoms(row.smiles), dtype=float)
            spin_dens.append(spin_densities_mol)
            dGs.append(0)
            BVs.append(0)
            fr_dGs.append(0)

    atom_desc_dataset = pd.DataFrame()
    atom_desc_dataset['smiles'] = df_pred['smiles']
    atom_desc_dataset['partial_charge'] = df_pred['charges_all_atom']
    atom_desc_dataset['spin_dens'] = spin_dens
    atom_desc_dataset['dG'] = dGs
    atom_desc_dataset['BV'] = BVs
    atom_desc_dataset['fr_dG'] = fr_dGs

    atom_values = []

    for row in df_mapped.itertuples():

        atom_values.append(extract_data(atom_desc_dataset, row.rxn_smiles))

    atom_values = pd.DataFrame(atom_values,
                                 columns=['dG', 'dG_forward', 'dG_reverse', 'q_reac0', 'qH_reac0',
                                          'q_reac1', 's_reac1', 'q_prod0', 's_prod0', 'q_prod1',
                                          'qH_prod1', 'BV_reac1', 'BV_prod0', 'fr_dG_forward', 'fr_dG_reverse'])

    rxns_desc_dataset = pd.concat([df_mapped, atom_values], axis=1)

    rxns_desc_dataset = rxns_desc_dataset.drop(columns=['rxn_id', 'DG_TS', 'G_r', 'DG_TS_tunn'])

    atom_desc_dataset.drop(columns=['dG'], inplace=True, axis=1)
    atom_desc_dataset.drop(columns=['BV'], inplace=True, axis=1)
    atom_desc_dataset.drop(columns=['fr_dG'], inplace=True, axis=1)
    rxns_desc_dataset.rename(columns={'rxn_smiles' : 'smiles'}, inplace=True)

    atom_desc_dataset.to_pickle('../../data/atom_desc_radicals.pkl')
    rxns_desc_dataset.to_pickle('../../data/rxn_desc_radicals.pkl')


def extract_data(atom_desc_dataset, rxn):

    r0, r1 = rxn.split(">>")[0].split(".")
    p0, p1 = rxn.split(">>")[1].split(".")

    # r1 is the radical of p1 and p0 is the radical of r0

    dG_forward = atom_desc_dataset.dG.loc[atom_desc_dataset.smiles == p0].values[0]
    dG_reverse = atom_desc_dataset.dG.loc[atom_desc_dataset.smiles == r1].values[0]
    fr_dG_forward = atom_desc_dataset.fr_dG.loc[atom_desc_dataset.smiles == p0].values[0]
    fr_dG_reverse = atom_desc_dataset.fr_dG.loc[atom_desc_dataset.smiles == r1].values[0]
    BV_reac1 = atom_desc_dataset.BV.loc[atom_desc_dataset.smiles == r1].values[0]
    BV_prod0 = atom_desc_dataset.BV.loc[atom_desc_dataset.smiles == p0].values[0]
    dG = dG_forward - dG_reverse

    os_idx_prod0, cs_idx_reac0, H_idx_reac0 = get_atom_idx(p0, r0)
    os_idx_reac1, cs_idx_prod1, H_idx_prod1 = get_atom_idx(r1, p1)

    qH_reac0 = atom_desc_dataset.partial_charge.loc[atom_desc_dataset.smiles == r0].values[0][H_idx_reac0]
    q_reac0 = atom_desc_dataset.partial_charge.loc[atom_desc_dataset.smiles == r0].values[0][cs_idx_reac0]

    qH_prod1 = atom_desc_dataset.partial_charge.loc[atom_desc_dataset.smiles == p1].values[0][H_idx_prod1]
    q_prod1 = atom_desc_dataset.partial_charge.loc[atom_desc_dataset.smiles == p1].values[0][cs_idx_prod1]

    q_reac1 = atom_desc_dataset.partial_charge.loc[atom_desc_dataset.smiles == r1].values[0][os_idx_reac1]
    s_reac1 = atom_desc_dataset.spin_dens.loc[atom_desc_dataset.smiles == r1].values[0][os_idx_reac1]

    q_prod0 = atom_desc_dataset.partial_charge.loc[atom_desc_dataset.smiles == p0].values[0][os_idx_prod0]
    s_prod0 = atom_desc_dataset.spin_dens.loc[atom_desc_dataset.smiles == p0].values[0][os_idx_prod0]

    data = (dG, dG_forward, dG_reverse, q_reac0, qH_reac0,
            q_reac1,  s_reac1, q_prod0, s_prod0, q_prod1,
            qH_prod1, BV_reac1, BV_prod0, fr_dG_forward, fr_dG_reverse)

    return data


def get_atom_idx(os_smi, cs_smi):

    os_mol = Chem.MolFromSmiles(os_smi)

    cs_mol = Chem.MolFromSmiles(cs_smi)
    cs_molH = Chem.AddHs(cs_mol)

    for atom in os_mol.GetAtoms():
        if atom.GetNumRadicalElectrons():
            os_atom_mapnum = atom.GetAtomMapNum()
            os_atom_idx = atom.GetIdx()
            break

    for atom in cs_molH.GetAtoms():
        if atom.GetAtomMapNum() == os_atom_mapnum:
            cs_atom_idx = atom.GetIdx()
            for ngb in atom.GetNeighbors():
                if ngb.GetSymbol() == 'H':
                    H_atom_idx = ngb.GetIdx()
                    break
            break

    return os_atom_idx, cs_atom_idx, H_atom_idx


def is_radical(smiles):

    mol = Chem.MolFromSmiles(smiles)

    return Chem.Descriptors.NumRadicalElectrons(mol)


def num_atoms(smiles):

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)

    return mol.GetNumAtoms()



if __name__ == '__main__':
    print("hello")
