import pandas as pd
from rdkit import Chem


def clean_data():
    """ Clean and modify the data from ACS Omega paper """

    data = pd.read_csv('../../data/data_omega/data_omega', delimiter=';')
    data.rename(columns={'Col 1': 'Substrate_Idx', 'Col 2': 'Mol.Rad', 'Col 3':  'BDFE', 'Col 4': 'G_rxn',
                         'Col 5': 'Electronegativity', 'Col 6': 'delta_Electronegativity_squared',
                         'Col 7': 'Buried_Volume', 'Col 8': 'G_act'}, inplace=True)

    pair_mol_rad = data['Mol.Rad'].tolist()

    all_smiles = []
    smi_rad = []
    smi_mol = []

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        mol = canonicalize_smiles(mol)
        rad = canonicalize_smiles(rad)
        smi_rad.append(rad)
        smi_mol.append(mol)
        all_smiles.append(rad)
        all_smiles.append(mol)

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

    [all_smiles.append(smi) for duo in set_rad for smi in duo]

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
    data['rxn_smiles'] = rxns
    data['mol'] = smi_mol
    data['rad'] = smi_rad
    data.drop(columns=['Mol.Rad'], inplace=True)

    data_autodE_format = pd.DataFrame()
    data_autodE_format['rxn_id'] = idx
    data_autodE_format['rxn_smiles'] = rxns
    data_autodE_format['G_r'] = data['G_rxn']
    data_autodE_format['DG_TS'] = data['G_act']
    data_autodE_format['DG_TS_tunn'] = data['G_act']

    data_autodE_format.to_csv('../../data/data_omega/reactivity_omega_database.csv')

    data.to_csv('../../data/data_omega/clean_data_omega.csv')
    data.to_pickle('../../data/data_omega/clean_data_omega.pkl')

    data_exp = pd.read_csv('../../data/data_omega/cumoexp.csv', delimiter=';')
    pair_mol_rad = data_exp['Mol.Rad'].tolist()

    smi_rad = []
    smi_mol = []
    rxns = []

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        mol = canonicalize_smiles(mol)
        rad = canonicalize_smiles(rad)
        smi_rad.append(rad)
        smi_mol.append(mol)
        rxns.append(f"{mol}.{cumo_rad}>>{rad}.{cumo_mol}")
        all_smiles.append(rad)
        all_smiles.append(mol)

    data_exp['rad'] = smi_rad
    data_exp['mol'] = smi_mol
    data_exp['rxn_smiles'] = rxns

    data_exp_autodE_format = pd.DataFrame()
    data_exp_autodE_format['rxn_id'] = data_exp.index
    data_exp_autodE_format['rxn_smiles'] = rxns
    data_exp_autodE_format['G_r'] = None
    data_exp_autodE_format['DG_TS'] = data_exp['gibbs']
    data_exp_autodE_format['DG_TS_1'] = data_exp['gibbs_exp']
    data_exp_autodE_format['DG_TS_2'] = data_exp['gibbs_relative']

    data_exp_autodE_format.to_csv('../../data/data_omega/reactivity_exp_omega_database.csv')
    data_exp.to_csv('../../data/data_omega/clean_data_omega_exp.csv')
    data_exp.to_pickle('../../data/data_omega/clean_data_omega_exp.pkl')

    data_selectivity = pd.read_csv('../../data/data_omega/selectivity.csv', delimiter=';')
    pair_mol_rad = data_selectivity['Mol.Rad'].tolist()

    smi_rad = []
    smi_mol = []
    rxns = []

    for smi in pair_mol_rad:
        mol, rad = smi.split('.')
        mol = canonicalize_smiles(mol)
        rad = canonicalize_smiles(rad)
        smi_rad.append(rad)
        smi_mol.append(mol)
        rxns.append(f"{mol}.{methoxyl_rad}>>{rad}.{methoxyl_mol}")
        all_smiles.append(rad)
        all_smiles.append(mol)

    data_selectivity['rad'] = smi_rad
    data_selectivity['mol'] = smi_mol
    data_selectivity['rxn_smiles'] = rxns

    data_selectivity_autodE_format = pd.DataFrame()
    data_selectivity_autodE_format['rxn_id'] = data_selectivity.index
    data_selectivity_autodE_format['rxn_smiles'] = rxns
    data_selectivity_autodE_format['G_r'] = None
    data_selectivity_autodE_format['DG_TS'] = data_selectivity['gibbs']
    data_selectivity_autodE_format['DG_TS_tunn'] = None

    data_selectivity_autodE_format.to_csv('../../data/data_omega/reactivity_selectivity_omega_database.csv')
    data_selectivity.to_csv('../../data/data_omega/clean_data_omega_selectivity.csv')
    data_selectivity.to_pickle('../../data/data_omega/clean_data_omega_selectivity.pkl')

    df_smiles = pd.DataFrame(all_smiles, columns=['smiles'])
    df_smiles.drop_duplicates(inplace=True)
    df_smiles.to_csv('../../data/data_omega/species_reactivity_omega_dataset.csv')

    return None


def canonicalize_smiles(smiles):

    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


if __name__ == '__main__':
    clean_data()
