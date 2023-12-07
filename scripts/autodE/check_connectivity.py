import os
import pandas as pd
from rdkit import Chem
import autode as ade


def main_function():

    pwd = os.getcwd()
    outputs_ade = '../../autodE_input/0.autode_resume_dir'

    data_folder = os.path.join(pwd, '../../data')
    outputs_dir = os.path.join(pwd, outputs_ade)

    df = pd.read_csv(f'{data_folder}/autode_results.csv', index_col=0)

    r0_list = []
    r1_list = []
    check = []

    for rxn in df.itertuples():
        os.chdir(outputs_dir)
        os.chdir(f"rxn_{rxn.ID:07}")
        ts_file = f"TS_{rxn.ID:07}.xyz"
        r0_file = f"r0_{rxn.ID:07}.xyz"
        r1_file = f"r1_{rxn.ID:07}.xyz"
        r0_smi, r1_smi, _, _ = reacs_prods(rxn.RXN_SMILES)
        ts_molecule = ade.Molecule(ts_file, mult=2)
        r0_molecule = ade.Molecule(r0_file, charge=charge(r0_smi))
        r1_molecule = ade.Molecule(r1_file, mult=2, charge=charge(r1_smi))
        atoms_r0 = check_connection(ts_molecule, r0_molecule, 0)
        atoms_r1 = check_connection(ts_molecule, r1_molecule, 1)
        r0_list.append(atoms_r0)
        r1_list.append(atoms_r1)
        if (len(atoms_r1) == 0 and len(atoms_r0) == 0) or (len(atoms_r1) == 1 and len(atoms_r0) == 2):
            check.append('no')
        else:
            check.append('yes')
        os.chdir(outputs_dir)

    df['r0_atoms'] = r0_list
    df['r1_atoms'] = r1_list
    df['check'] = check
    #df.to_csv(f'{data_folder}/df_connectivity.csv')
    df_checked = df.loc[df['check'] == 'yes']
    df_checked.to_csv(f'{data_folder}/df_conn_check.csv')


def charge(smi):

    mol = Chem.MolFromSmiles(smi)
    mol_H = Chem.AddHs(mol)
    mol_H.ComputeGasteigerCharges()
    contribs = [mol_H.GetAtomWithIdx(i).GetDoubleProp('_GasteigerCharge')
                for i in range(mol_H.GetNumAtoms())]
    charge = round(sum(contribs))

    return charge


def check_connection(ts_molecule, molecule, reactant):

    n_atoms_mol = molecule.n_atoms
    if reactant:
        n_atoms_ts = ts_molecule.n_atoms
        first_atom = n_atoms_ts - n_atoms_mol
        last_atom = n_atoms_ts
    else:
        first_atom = 0
        last_atom = n_atoms_mol

    edges_TS = ts_molecule.graph.edges()
    edges_mol = molecule.graph.edges()
    atoms = []

    for atom in range(first_atom, last_atom):
        atom_edges_TS = sorted(list(edges_TS([atom])))
        atom_edges_mol = sorted(list(edges_mol([atom - first_atom])))
        atom_edges_mol = [(edge[0] + first_atom, edge[1] + first_atom) for edge in atom_edges_mol]
        if atom_edges_TS != atom_edges_mol:
            atoms.append(atom)

    return atoms


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


def obtain_molecule(smiles, addH = True):
    """ Return a molecule for the given SMILES """

    molecule = Chem.MolFromSmiles(smiles)
    if addH:
        molecule = Chem.rdmolops.AddHs(molecule)
    Chem.Kekulize(molecule, clearAromaticFlags=True)

    return molecule

if __name__ == "__main__":
    main_function()

    