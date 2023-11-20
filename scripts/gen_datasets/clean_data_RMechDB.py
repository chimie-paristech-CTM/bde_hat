import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors


elem_list = ['C', 'O', 'N', 'F', 'Br', 'Cl', 'S', 'I', 'P', 'B', 'H']
no_paton_data = ["P", "S", "Cl", "Br", "I", "F", "B"]


def filter_data():

    df = pd.read_csv('original_data/all_modified_rmechdb.csv', sep=';')

    df_abs = df.loc[df['class_II'] == 'abstraction']
    df_abs = df_abs.loc[df_abs['condition'] == 'Room Temperature']
    df_abs = df_abs.loc[df_abs['class_I'] == 'Propagation']
    df_abs['two_reacs'] = df_abs['SMIRKS'].apply(lambda x: len(x.split('>>')[0].split('.')) == 2)
    df_abs = df_abs.loc[df_abs['two_reacs']]
    df_abs['one_rad'] = df_abs['SMIRKS'].apply(lambda x: just_one_radical(x))
    df_abs = df_abs.loc[df_abs['one_rad']]
    df_abs['not_spectator'] = df_abs['SMIRKS'].apply(lambda x: '\\' not in x)
    df_abs = df_abs.loc[df_abs['not_spectator']]
    df_abs['SMILES'] = df_abs['SMIRKS'].apply(lambda x: remove_atom_mapping(x))
    df_abs['HAT'] = df_abs['SMILES'].apply(lambda x: is_HAT(x))
    df_abs = df_abs.loc[df_abs['HAT']]
    df_abs['elem_surrogate'] = df_abs['SMILES'].apply(lambda x: elem_surrogate(x))
    df_abs = df_abs.loc[df_abs['elem_surrogate']]
    df_abs['no_paton_data'] = df_abs['SMILES'].apply(lambda x: get_elem_no_paton_data(x))
    df_abs = df_abs.drop_duplicates(subset=['SMILES'])
    df_abs.reset_index(inplace=True)
    df_abs.rename(columns={'SMILES': 'rxn_smiles', 'index': 'rxn_id'}, inplace=True)
    df_abs['specific_rxn'] = df_abs['rxn_smiles'].apply(lambda x: filter_rxns(x))
    df_abs['formal_charges'] = df_abs['rxn_smiles'].apply(lambda x: formal_charges(x))

    df_abs.to_csv('RMechDB_clean.csv')


def formal_charges(rxn):

    r0, r1, _, _ = reacs_prods(rxn)

    for smi in [r0, r1]:
        mol = Chem.MolFromSmiles(smi)
        for atom in mol.GetAtoms():
            if atom.GetFormalCharge():
                return True

    return False


def filter_rxns(rxn):

    species = reacs_prods(rxn)

    sum = 0

    for specie in species:
        mol = Chem.MolFromSmiles(specie)

        for atom in mol.GetAtoms():
            if atom.GetSymbol() in no_paton_data and mol.GetNumAtoms() == 1:
                sum = sum + 2
            elif atom.GetSymbol() in no_paton_data and mol.GetNumAtoms() != 1:
                sum = sum + 3

    if sum in [0, 4]:
        return True
    else:
        return False


def counting(df_abs):

    p = 0
    s = 0
    cl = 0
    br = 0
    i = 0

    for row in df_abs.itertuples():
        species = reacs_prods(row.SMILES)

        for specie in species[:2]:
            mol = Chem.MolFromSmiles(specie)

            for atom in mol.GetAtoms():
                if atom.GetSymbol() in no_paton_data:
                    match atom.GetSymbol():
                        case 'P':
                            p += 1
                        case 'S':
                            s += 1
                        case 'Cl':
                            cl += 1
                        case 'Br':
                            br += 1
                        case 'I':
                            i += 1

    with open('counting_upper_row.txt', 'w') as file:
        file.write(f'P: {p}  {p*100/702}\n')
        file.write(f'S: {s} {s*100/702}\n')
        file.write(f'Cl: {cl} {cl*100/702}\n')
        file.write(f'Br: {br} {br*100/702}\n')
        file.write(f'I: {i} {i*100/702}\n')


def get_elem_no_paton_data(rxn):

    species = reacs_prods(rxn)

    for specie in species[:2]:
        mol = Chem.MolFromSmiles(specie)

        for atom in mol.GetAtoms():
            if atom.GetSymbol() in no_paton_data:
                return True

    return False


def elem_surrogate(rxn):

    species = reacs_prods(rxn)

    for specie in species[:2]:
        mol = Chem.MolFromSmiles(specie)

        for atom in mol.GetAtoms():
            if atom.GetSymbol() not in elem_list:
                return False

    return True


def is_HAT(rxn):
    r0, _, p0, _ = reacs_prods(rxn)

    r0_mol = Chem.MolFromSmiles(r0)
    p0_mol = Chem.MolFromSmiles(p0)

    for atom in p0_mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            rad_idx = atom.GetIdx()
            break

    substruct = p0_mol.GetSubstructMatch(r0_mol)

    if not substruct:
        return False

    mol_idx = substruct.index(rad_idx)

    r0_mol = Chem.AddHs(r0_mol)
    p0_mol = Chem.AddHs(p0_mol)

    mol_ngh = [ngh.GetSymbol() for ngh in r0_mol.GetAtomWithIdx(mol_idx).GetNeighbors()]
    rad_ngh = [ngh.GetSymbol() for ngh in p0_mol.GetAtomWithIdx(rad_idx).GetNeighbors()]

    mol_ngh_set = set(mol_ngh)
    rad_ngh_set = set(rad_ngh)

    if len(mol_ngh_set) == len(rad_ngh_set):
        for elem in mol_ngh_set:
            repeat_elem_mol = mol_ngh.count(elem)
            repeat_elem_rad = rad_ngh.count(elem)
            if (repeat_elem_rad - repeat_elem_mol != 0):
                return True if elem == 'H' else False
    else:
        elem = mol_ngh_set - rad_ngh_set
        return True if 'H' in elem else False


def remove_atom_mapping(smiles):
    species = reacs_prods(smiles)
    rad_smi = []
    mol_smi = []
    for smi in species:
        mol = Chem.MolFromSmiles(smi)
        [atom.SetAtomMapNum(0) for atom in mol.GetAtoms()]
        if Chem.Descriptors.NumRadicalElectrons(mol):
            rad_smi.append(Chem.MolToSmiles(mol))
        else:
            mol_smi.append(Chem.MolToSmiles(mol))
    rxn = f"{mol_smi[0]}.{rad_smi[0]}>>{rad_smi[1]}.{mol_smi[1]}"

    return rxn


def reacs_prods(smiles_rxn):
    """ Return the components of the rxn """

    dot_index = smiles_rxn.index('.')
    sdot_index = smiles_rxn.index('.', dot_index + 1)
    limit_index = smiles_rxn.index('>>')
    reac1 = smiles_rxn[:dot_index]
    reac2 = smiles_rxn[dot_index + 1: limit_index]
    prod1 = smiles_rxn[limit_index + 2: sdot_index]
    prod2 = smiles_rxn[sdot_index + 1:]

    return reac1, reac2, prod1, prod2


def just_one_radical(smiles):

    r0, r1 = smiles.split('>>')[0].split('.')
    mol_r0 = Chem.MolFromSmiles(r0)
    mol_r1 = Chem.MolFromSmiles(r1)

    if (Chem.Descriptors.NumRadicalElectrons(mol_r0) + Chem.Descriptors.NumRadicalElectrons(mol_r1)) == 1:
        return True
    else:
        return False

