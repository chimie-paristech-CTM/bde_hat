import pandas as pd
from rdkit import Chem


def clean_data_training():

    df = pd.read_csv('../../data/TrainingSet-2926-SMILES.csv')

    rxn_smiles = []
    filter_rxn = []

    df = df[~df.isna().any(axis=1)]  # remove 2 nan elements

    for row in df.itertuples():
        if 'nan' in row:
            filter_rxn.append(False)
            rxn_smiles.append(False)
            continue
        filter_rxn.append(filtering(row))
        rxn_smiles.append(f"{row.substrate_reactant}.{row.radical_reactant}>>{row.substrate_product}.{row.radical_product}")

    df['filter'] = filter_rxn
    df['RXN_SMILES'] = rxn_smiles
    df.rename(columns={'Barrier': 'DG_TS'}, inplace=True)
    df = df.loc[~df['filter']]

    df.to_csv('../../data/training_hong_clean.csv')


def filtering(row):

    mol1 = Chem.MolFromSmiles(row.substrate_reactant)
    rad2 = Chem.MolFromSmiles(row.radical_reactant)
    mol2 = Chem.MolFromSmiles(row.substrate_product)
    rad1 = Chem.MolFromSmiles(row.radical_product)

    mols = [mol1, rad2, mol2, rad1]

    for mol in mols:
        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in ['C', 'N', 'O']:
                return True
            if atom.GetFormalCharge() != 0:
                return True

    return False


if __name__ == "__main__":
    clean_data_training()
