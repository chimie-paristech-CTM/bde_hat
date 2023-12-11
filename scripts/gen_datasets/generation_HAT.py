#!/usr/bin/python
import pandas as pd
import numpy as np
import time
import rdkit
import multiprocessing
import warnings
from rdkit import Chem
from rdkit import RDLogger
from bde.fragment import fragment_iterator
from drfp import DrfpEncoder
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from ast import literal_eval
from scipy.spatial.distance import cosine as cosine_distance


# Ignore rdkit warnings: https://github.com/rdkit/rdkit/issues/2683
RDLogger.DisableLog('rdApp.*')

# Ignore pandas warning
warnings.simplefilter(action='ignore', category=FutureWarning)

""" Here, 20200415_radical_database.csv has been downloaded from the figshare: https://doi.org/10.6084/m9.figshare.c.4944855.v1 """
DB = pd.read_csv('../data/20200415_radical_database.csv', index_col = 0)
H_FreeEnergy = DB.loc[DB.SMILES == '[H]'].FreeEnergy.values[0]


def make_reactions():
    """ Main function """
    
    t0 = time.time()

    # Generating the inital dataset using Paton's scripts
    data = preparing_dataset()

    # Generating two random sets of 1 000 000 samples
    t1 = time.time()
    data1 = data.sample(n=1000000, replace=True, random_state=3142, ignore_index=True)
    data2 = data.sample(n=1000000, replace=True, random_state=2718, ignore_index=True)

    rxns_list = []
    for react1, react2 in zip(data1.itertuples(), data2.itertuples()):
        if react1 == react2:
            continue
        rxn = f"{react1.molecule}.{get_frag(react2)}>>{get_frag(react1)}.{react2.molecule}"
        rxns_list.append([rxn, react1.bond_index, react2.bond_index, react1.bond_type, react2.bond_type])
    t2 = time.time()

    rxns_1M = pd.DataFrame(rxns_list, columns=['rxn', 'bond_index_broken', 'bond_index_formed', 'bond_type_broken', 'bond_type_formed'])
    rxns_1M.to_csv('../data/reactions.csv')  

    frequency = rxns_1M.groupby(['bond_type_broken', 'bond_type_formed']).size()

    with open('info.txt', 'a') as file:
        file.write(f"\nTime to combine reactions: {t2 - t1} seconds")
        file.write(f"\nThe combination of broken_bond and formed_bond:\n {frequency}")

    complementary_info = multiprocess_func(info, rxns_1M.iloc, 70)

    # Creating the full dataset
    rxns_1M = pd.concat([rxns_1M, pd.DataFrame(complementary_info[0])], axis=1)
    rxns_1M.to_csv('../data/reactions_1M.csv')

    with open('info.txt', 'a') as file:
        file.write(f"\nErrors during info function: {complementary_info[1]}")

    # Subset for benchmark keyword 'hmethod'
    final_subset_xtb_g16_autodE(rxns_1M)

    # Subset for autodE
    rxns_2k = iterative_random_sampling(random_samples_initial=2000, cutoff=0.85)
    rxns_2k_final = rxns_1M.loc[rxns_2k.index]
    rxns_2k_final.to_csv('../data/reactions_2k.csv')

    # Fixing some possibles problems with the stereochemistry
    rxns_2k_autodE = final_subset_autodE(rxns_2k_final)
    rxns_2k_autodE.to_csv('../data/reactions_2k_autodE.csv')

    t3 = time.time()

    with open('info.txt', 'a') as file:
        file.write(f"\nTime to run entire script: {t3 - t0} seconds")
    
    return None


def preparing_dataset():

    start = time.time()
    
    db_mol = DB.loc[DB.type == 'molecule']

    reaction_db = pd.DataFrame()
    
    for molecule in db_mol.itertuples():
        reaction_db = pd.concat([reaction_db, pd.DataFrame(fragment_iterator(molecule.SMILES, skip_warnings=True))], ignore_index=True)
    
    reaction_db = reaction_db.drop('is_valid_stereo', axis=1)
    reaction_db = reaction_db.loc[reaction_db.bond_type.isin(['C-H', 'H-N', 'H-O'])]
    
    original_shape = reaction_db.shape
    reaction_db = reaction_db[reaction_db.fragment2.isin(DB.SMILES)]
    reaction_db = reaction_db[reaction_db.molecule.isin(DB.SMILES)]
    reaction_db = reaction_db[reaction_db.fragment1.isin(DB.SMILES)]
    new_shape = reaction_db.shape
    missing = original_shape[0] - new_shape[0]

    reaction_db = reaction_db.drop_duplicates(subset=['molecule', 'bond_type', 'fragment1', 'fragment2'], ignore_index=True)
    reaction_db.to_csv('../data/reactions_db_wo_duplicates.csv')

    end = time.time()
    with open('info.txt', 'w') as file:
        file.write(f"\nFragments that are not in the original DB: {missing}")
        file.write(f"\nTime for creating initial dataset: {end - start} seconds")

    return reaction_db 


def encode(smiles_rxn, mapping_bool=False):
    """ Encode a smiles rxn """

    return DrfpEncoder.encode(smiles_rxn, mapping=mapping_bool)


def multiprocess_func(func, elements, num_cpu=multiprocessing.cpu_count(), verbose=True):
    """ This function will run the func in multiples cpus """

    invalid_temp = 0
    outputs = []
    start_time = time.time()
    with ProcessPool(max_workers=num_cpu) as pool:
        future = pool.map(func, elements, timeout=100)
        iterator = future.result()
        while True:
            try:
                result = next(iterator)
                if result is not None:
                    outputs.append(result)
            except StopIteration:
                break
            except TimeoutError as error:
                invalid_temp += 1
            except ValueError as error:
                invalid_temp += 1
                raise
            except:
                pass
        pool.close()
        pool.join()
        end_time = time.time()

        if verbose:

            with open("info.txt", "a") as file:
                file.write(f"\nThe {func} last {end_time - start_time} seconds")
        
    return (outputs, invalid_temp)


def info(rxn_data):
    """ From a smile rxn, obtain charges, spin densities and free Gibbs energies"""

    r1, r2, p1, p2 = reacs_prods(rxn_data.rxn)

    charges_reac1 = get_charges_molecule(r1, int(rxn_data.bond_index_broken))
    charge_spin_prod1 = get_charge_spin_radical(p1)
    charges_prod2 = get_charges_molecule(p2, int(rxn_data.bond_index_formed))
    charge_spin_reac2 = get_charge_spin_radical(r2)
    reac2_atomspins = DB.loc[DB.SMILES == r2].AtomSpins.values
    prod1_atomspins = DB.loc[DB.SMILES == p1].AtomSpins.values
    reac1_atomcharges = DB.loc[DB.SMILES == r1].AtomCharges.values
    reac2_atomcharges = DB.loc[DB.SMILES == r2].AtomCharges.values
    prod1_atomcharges = DB.loc[DB.SMILES == p1].AtomCharges.values
    prod2_atomcharges = DB.loc[DB.SMILES == p2].AtomCharges.values
    dG_forward = (((DB.loc[DB.SMILES == p1].FreeEnergy.values + H_FreeEnergy)-
                  (DB.loc[DB.SMILES == r1].FreeEnergy.values)) * 627.509)
    dG_reverse = (((DB.loc[DB.SMILES == r2].FreeEnergy.values + H_FreeEnergy)-
                  (DB.loc[DB.SMILES == p2].FreeEnergy.values)) * 627.509)
    dG = dG_forward - dG_reverse
    return {
        "charges_reac1": charges_reac1,
        "charge_spin_prod1": charge_spin_prod1,
        "charges_prod2": charges_prod2,
        "charge_spin_reac2": charge_spin_reac2,
        "reac2_atomspins": reac2_atomspins,
        "prod1_atomspins": prod1_atomspins,
        "reac1_atomcharges": reac1_atomcharges,
        "reac2_atomcharges": reac2_atomcharges,
        "prod1_atomcharges": prod1_atomcharges,
        "prod2_atomcharges": prod2_atomcharges,
        "dG": dG,
        "dG_forward": dG_forward,
        "dG_reverse": dG_reverse,
        }


def get_charge_spin_radical(smiles):
    """ Return the charge and the spin density of the radical atom """

    molecule = obtain_molecule(smiles)
    AtomRadicalIdx = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetNumRadicalElectrons() == 1]
    AtomCharge = literal_eval(DB.loc[DB.SMILES == smiles].AtomCharges.values[0])[AtomRadicalIdx[0]]
    AtomSpin = literal_eval(DB.loc[DB.SMILES == smiles].AtomSpins.values[0])[AtomRadicalIdx[0]]

    return (AtomCharge, AtomSpin)


def get_charges_molecule(smiles, bond_index):
    """ Return the charge of the atoms of the bond involved in the reaction """

    molecule = obtain_molecule(smiles)
    bond = molecule.GetBondWithIdx(bond_index)
    beginAtomIdx = bond.GetBeginAtomIdx()
    endAtomIdx = bond.GetEndAtomIdx()
    AtomCharges = str({
        bond.GetBeginAtom().GetSymbol(): literal_eval(DB.loc[DB.SMILES == smiles].AtomCharges.values[0])[beginAtomIdx],
        bond.GetEndAtom().GetSymbol(): literal_eval(DB.loc[DB.SMILES == smiles].AtomCharges.values[0])[endAtomIdx]
    })

    return AtomCharges


def get_frag(rxn_info : pd.Series) -> str:
    """ Return the fragment that is != [H] """

    return rxn_info.fragment2 if rxn_info.fragment2 != '[H]' else rxn_info.fragment1


def iterative_random_sampling(random_samples_initial=2000, cutoff=0.85):
    """ This function takes a random samples of rxns and filters until all the rxns are above a specific treshold """

    start = time.time()
    rxns_space = pd.read_csv('../data/reactions_1M.csv', index_col=0, usecols=[0,1,4,5])

    # create the final dataset of reactions, the temporal dataset of reactions and the final dataset with the fingerprints.
    rxns = pd.DataFrame()
    rxns_drfp = pd.DataFrame()
    rxns_temp = rxns_space.sample(n=random_samples_initial, random_state=6023)
    rxns = pd.concat([rxns, rxns_temp])
    steps = 0
    rxns_explored = random_samples_initial
    fixed_rxns = 0

    while True:

        # Drop the temporal_reactions from the initial dataset of 1 million
        rxns_temp_idx = [rxns_space[rxns_space.rxn == i.rxn].index[0] for i in rxns_temp.itertuples()]
        rxns_space = rxns_space.drop(labels = rxns_temp_idx)

        # Encode the reactions
        fps, errors = multiprocess_func(encode, rxns_temp.rxn.tolist(), 70, verbose=False)
        rxns_temp_drfp = pd.DataFrame(np.array([item for s in fps for item in s]))

        # Add the temp encoded reactions to the final dataset of fingerprints
        rxns_drfp = pd.concat([rxns_drfp, rxns_temp_drfp], ignore_index=True)
        

        # Elements that for the cutoff should be dropped
        to_remove = []

        # Calculating all the distances ... squared matrix ... but I only need a half
        # if an element, will be removed, it is not necesary to calculate the distance
        # and for the elements that have passed to the next step, it is also not necesary to calculate the distance
        for i in range(len(rxns)):
            if i in to_remove:
                continue
            for j in range(fixed_rxns, len(rxns)):
                if (j > i) & (j not in to_remove):
                    cos_d = cosine_distance(rxns_drfp.iloc[i].values, rxns_drfp.iloc[j].values)
                    if cos_d < cutoff:
                        to_remove.append(j)

        # Eliminating duplicated elements
        to_remove = set(to_remove)
        to_remove_rxns = [rxns.iloc[i].name for i in to_remove]

        # Eliminating reactions
        rxns_drfp = rxns_drfp.drop(labels=to_remove)
        rxns = rxns.drop(to_remove_rxns)

        # Check if I need to take more samples
        fixed_rxns = len(rxns)
        random_samples = random_samples_initial - fixed_rxns

        if random_samples == 0:
            message = f"All the reactions are above {cutoff}."
            break
        if random_samples > len(rxns_space):
            message = f"Just could find {len(rxns)} that are above {cutoff}."
            break

        rxns_temp = rxns_space.sample(n=random_samples, random_state=6023)
        rxns = pd.concat([rxns, rxns_temp])
        steps += 1
        rxns_explored += random_samples
    
    end = time.time()
    frequency = rxns.groupby(['bond_type_broken', 'bond_type_formed']).size()

    with open('info.txt', 'a') as file:
        file.write(f"\n{message}")
        file.write(f"\nTime for random sampling iteratively: {end - start} seconds.")
        file.write(f"\nrxn iterated: {rxns_explored}")
        file.write(f"\nSteps: {steps}.")
        file.write(f"\nThe combination of broken_bond and formed_bond in the 3k subset:\n {frequency}")

    return rxns


def reacs_prods(smiles_rxn):
    """ Return the components of the rxn """

    dot_index   = smiles_rxn.index('.')
    sdot_index  = smiles_rxn.index('.', dot_index + 1)
    limit_index = smiles_rxn.index('>>')
    reac1 = smiles_rxn[:dot_index]
    reac2 = smiles_rxn[dot_index + 1: limit_index]
    prod1 = smiles_rxn[limit_index + 2: sdot_index]
    prod2 = smiles_rxn[sdot_index + 1:]

    return reac1, reac2, prod1, prod2


def obtain_molecule(smiles, addH=True):
    """ Return a molecule for the given SMILES """

    molecule = rdkit.Chem.MolFromSmiles(smiles)
    if addH:
        molecule = rdkit.Chem.rdmolops.AddHs(molecule)
    rdkit.Chem.Kekulize(molecule, clearAromaticFlags=True)

    return molecule


def final_subset_autodE(df):
    """ Create the subset for autodE calculations """

    flip_rxns = stereochemistry_rxn(df)
    flip_rxns_complementary = [info(rxn) for rxn in flip_rxns.iloc]
    flip_rxns = pd.concat([flip_rxns, pd.DataFrame(flip_rxns_complementary)], axis=1)

    to_remove = flip_rxns["index"] - 1000000

    rxns = df.drop(to_remove)

    flip_rxns = flip_rxns.set_index('index')

    final_subset = pd.concat([rxns, flip_rxns], axis=0)

    return final_subset


def stereochemistry_rxn(df):
    """ The point behind this ...

        there are reactions where a new stereocenter is formed, but the problem is that autodE doesn't ensure that the TS is leading to this stereoisomer.

        The general scheme for the rxns in the dataset is:

        r1 + r2 --> p1 + p2

        where r2 and p1 are radicals and r1 and p2 are closed shell molecules.

        We should check for p2 ... if in p2 is formed a new stereocenter, we just flip the reaction, but

        we can two cases, once that we flip the rxn:

        p1 + p2 --> r1 + r2

        case 1: in a r1 does not form a new stereocenter, everything is fine, just flip the rxn,

        case 2: but if in r1 is formed a new stereocenter, we will be in the same place and we will have to take a post_processing step

        case 1 will be saved in flip_rxns, and fixed before the autodE calculations and the case 2 will be saved in check_TS and print in a txt for later

        as we flip the rxn, now the bond_index_formed is the bond_index_broken and so on ...
        """

    check_TS = []
    flip_rxns = []

    for rxn in df.itertuples():

        r1, r2, p1, p2 = reacs_prods(rxn.rxn)

        if check_stereocenter(r1, rxn.bond_index_broken) and check_stereocenter(p2, rxn.bond_index_formed):
            check_TS.append(rxn.Index)
        elif not check_stereocenter(r1, rxn.bond_index_broken) and check_stereocenter(p2, rxn.bond_index_formed):
            rxn_smile = f"{p2}.{p1}>>{r2}.{r1}"
            flip_rxns.append((rxn.Index + 1000000, rxn_smile, rxn.bond_index_formed, rxn.bond_index_broken,
                              rxn.bond_type_formed, rxn.bond_type_broken))

    with open('check_TS_stereochemistry.txt', 'a') as file:
        file.write('The rxns that we have to check the TS:')
        [file.write(f"\n{ts}") for ts in check_TS]

    return pd.DataFrame(flip_rxns,
                        columns=['index', 'reactions', 'bond_index_broken', 'bond_index_formed', 'bond_type_broken',
                                 'bond_type_formed'])


def check_stereocenter(smiles, bond_index):
    """ Check if the forming/breaking bond is part of a stereocenter """

    mol = obtain_molecule(smiles)

    stereo_info = Chem.FindMolChiralCenters(mol, force=True, includeUnassigned=True, useLegacyImplementation=True)
    stereo_atoms = [elem[0] for elem in stereo_info]

    bond = mol.GetBondWithIdx(bond_index)
    atom1 = bond.GetBeginAtomIdx()
    atom2 = bond.GetEndAtomIdx()

    return True if (atom1 in stereo_atoms or atom2 in stereo_atoms) else False


def final_subset_xtb_g16_autodE(df):
    subset_30 = df.sample(n=30, random_state=100797)

    subset_30.to_csv('../data/subset_30_g16_xtb.csv')

    flip_rxns = stereochemistry_rxn(subset_30)
    flip_rxns_complementary = [info(rxn) for rxn in flip_rxns.iloc]
    flip_rxns = pd.concat([flip_rxns, pd.DataFrame(flip_rxns_complementary)], axis=1)

    to_remove = flip_rxns["index"] - 1000000

    rxns = subset_30.drop(to_remove)

    flip_rxns = flip_rxns.set_index('index')

    final_subset = pd.concat([rxns, flip_rxns], axis=0)

    final_subset.to_csv('../data/subset_30_g16_xtb_autodE.csv')

    return None


if __name__ == "__main__":
        make_reactions()
