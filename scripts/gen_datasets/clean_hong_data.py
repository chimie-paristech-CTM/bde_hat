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


def clean_data_test():

    df = pd.read_csv('../../data/SampleTest-ExpCH.csv')

    rad = ['O1', 'O2', 'O4', 'O6', 'N4']
    subs = ['S11', 'S26', 'S28', 'S37', 'S83', 'S17', 'S18', 'S19', 'S20', 'S40', 'S41', 'S43', 'S49', 'S82']

    df.drop(index=[0, 1], axis=0, inplace=True)
    df.reset_index(inplace=True)
    df['index'] = df['index'].apply(lambda x: x - 2)

    df = df[~df.isna().any(axis=1)]  # remove 2 nan elements

    df['filter'] = df['file_name'].apply(lambda x: True if x.split('-')[0] not in subs or x.split('-')[1] not in rad else False)
    df = df.loc[~df['filter']]

    df.to_csv('../../data/test_hong_clean.csv')

    generate_smiles_test(df)


def generate_smiles_test(df):

    subs = {'S11-P1': 'C1CCOC1.C2C[CH]OC2', 'S11-P2': 'C1CCOC1.C2[CH]COC2', 'S17-P1': 'CCOCC.CCO[CH]C',
            'S17-P2': 'CCOCC.CCOC[CH2]', 'S18-P1': 'C1OCOC1.C2O[CH]OC2', 'S18-P2': 'C1OCOC1.[CH]2OCOC2',
            'S19-P1': 'COCCOC.COC[CH]OC', 'S19-P2': 'COCCOC.COCCO[CH2]', 'S20-P1': 'CC(C)CCC(C)OC(C1=CC=CC=C1)=O.C[C](C)CCC(C)OC(C2=CC=CC=C2)=O',
            'S26-P1': 'OCCCCCN.OCCCC[CH]N', 'S26-P2': 'OCCCCCN.O[CH]CCCCN', 'S28-P1': 'CC(N(C)C)=O.[CH2]C(N(C)C)=O',
            'S28-P2': 'CC(N(C)C)=O.CC(N([CH2])C)=O', 'S37-P1': 'OCC1=CC2=CC=CC=C2C=C1.O[CH]C3=CC4=CC=CC=C4C=C3',
            'S40-P1': 'O=CC(C=C1)=CC=C1C2=CC=CC=C2.O=[C]C(C=C3)=CC=C3C4=CC=CC=C4', 'S41-P1': 'O=CC1=CC=CC=C1C.O=[C]C2=CC=CC=C2C',
            'S43-P1': 'O=CC1=CC2=CC=CC=C2C=C1.O=[C]C3=CC4=CC=CC=C4C=C3', 'S49-P1': 'O=CC1=CCCCC1.O=[C]C2=CCCCC2',
            'S82-P1': 'O=C1N([C@H]2CC[C@H](C)CC2)C(C3=C1C=CC=C3)=O.O=C4N([C@H]5CC[C@H](C)[CH]C5)C(C6=C4C=CC=C6)=O',
            'S82-P2': 'O=C1N([C@H]2CC[C@H](C)CC2)C(C3=C1C=CC=C3)=O.O=C4N([C@H]5CC[C@H]([CH2])CC5)C(C6=C4C=CC=C6)=O',
            'S83-P1': 'C[C@@]12[C@](CC[C@]3(C)[C@@H]2CCO3)([H])C(C)(C)CCC1.C[C@@]45[C@](CC[C@]6(C)[C@@H]5C[CH]O6)([H])C(C)(C)CCC4'}

    rads = {'O1': 'CC(O)(C)C1=CC=CC=C1.CC([O])(C)C2=CC=CC=C2', 'O2': 'O=C1N(O)C(C2=CC=CC=C21)=O.O=C3N([O])C(C4=CC=CC=C43)=O',
           'O4': 'OC(C)(C)C.[O]C(C)(C)C', 'O6': 'O=C(O)C1=CC=CC=C1.O=C([O])C2=CC=CC=C2', 'N4': 'O=C(NC)C1=CC=CC=C1.O=C([N]C)C2=CC=CC=C2'}

    rxn_smiles = []

    for row in df.itertuples():
        rad = row.file_name.split('-')[1]
        sub = row.file_name.split('-')[0] + '-' + row.file_name.split('-')[2]
        mol1, rad2 = subs[sub].split('.')
        mol2, rad1 = rads[rad].split('.')
        rxn = f"{canonical_smiles(mol1)}.{canonical_smiles(rad1)}>>{canonical_smiles(rad2)}.{canonical_smiles(mol2)}"
        rxn_smiles.append(rxn)

    df_smiles = pd.DataFrame()
    df_smiles['RXN_SMILES'] = rxn_smiles
    df_smiles['rxn_id'] = df['index'].tolist()
    df_smiles['DG_TS'] = df['Barrier'].tolist()

    df_smiles.to_csv('../../data/test_smiles_hong.csv')


def canonical_smiles(smi):

    mol = Chem.MolFromSmiles(smi)

    return Chem.MolToSmiles(mol)


if __name__ == "__main__":
    clean_data_training()
    clean_data_test()
