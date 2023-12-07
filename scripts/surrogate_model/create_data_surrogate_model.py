#!/usr/bin/python
import logging
import pandas as pd
from ast import literal_eval
import time
import multiprocessing
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from rdkit import Chem

DB = pd.read_csv('../../data/20200415_radical_database.csv', index_col=0)
H_FreeEnergy = DB.loc[DB.SMILES == '[H]'].FreeEnergy.values[0]


def create_train_test_surrogate_model():
    """
        main function
        we know the optimal parameters because the grid search
        we will train the model with that parameters
        in the test set will be all the molecules and radicals that are presented in the reactivity database
        and also the stereoisomer/diastereoisomer
    """

    db_reactivity = pd.read_csv('../../data/autode_results_clean.csv', index_col=0)
    db_reactivity.drop(columns=['time', 'frequency'], inplace=True)
    final_db_reactivity = pd.read_csv('../../data/autode_results_new_rxns_ts.csv', index_col=0)
    final_db_reactivity.ID = final_db_reactivity.ID + 2000000
    finished_rxns = pd.read_csv('../../data/reactivity_database_mapped.csv', index_col=0)
    final_db_reactivity = final_db_reactivity.loc[final_db_reactivity['ID'].isin(finished_rxns['rxn_id'])]
    final_db_reactivity = pd.concat([db_reactivity, final_db_reactivity], ignore_index=True)
    df_bv = pd.read_csv('../../data/df_buried_vol.csv', index_col=0)

    rxns = final_db_reactivity.RXN_SMILES.tolist()
    smiles = [map_mol(smi) for rxn in rxns for smi in reacs_prods(rxn)]
    smiles = list(set(smiles))

    paton_rxns = pd.read_csv('../../data/paton_rxns_frozen_BDE.csv')

    rxns_info = multiprocess_func(info_1, paton_rxns.iloc, 70)
    radicals = pd.DataFrame(rxns_info[0])
    radicals['Buried_Vol'] = radicals['smiles'].apply(lambda x: df_bv.loc[df_bv['smiles'] == x].Buried_Vol.values[0])
    radicals["Buried_Vol"] = radicals["Buried_Vol"].apply(lambda x: [x])

    molecules_DB = DB.loc[DB.type == 'molecule']

    molecules_DB = molecules_DB.drop(
        columns=["Enthalpy", "FreeEnergy", "SCFEnergy", "RotConstants", "VibFreqs", "IRIntensity", "mol", "type",
                 "Name", "AtomSpins"])

    molecules_DB = molecules_DB.rename(columns={"SMILES": "smiles"})
    molecules_DB = molecules_DB.rename(columns={"AtomCharges": "charges_all_atom"})
    molecules_DB["charges_all_atom"] = molecules_DB["charges_all_atom"].apply(literal_eval)

    spin_densities_cs = []
    for row in molecules_DB.itertuples():
        len_atoms = len(row.charges_all_atom)
        spin_densities_cs.append([0.] * len_atoms)

    molecules_DB["spin_densities"] = spin_densities_cs
    molecules_DB["dG"] = [0.] * len(spin_densities_cs)
    molecules_DB["frozen_dG"] = [0.] * len(spin_densities_cs)
    molecules_DB["Buried_Vol"] = [0.] * len(spin_densities_cs)
    molecules_DB["dG"] = molecules_DB["dG"].apply(lambda x: [x])
    molecules_DB["frozen_dG"] = molecules_DB["frozen_dG"].apply(lambda x: [x])
    molecules_DB["Buried_Vol"] = molecules_DB["Buried_Vol"].apply(lambda x: [x])

    final_data = pd.concat([radicals, molecules_DB], ignore_index=True)
    final_data['smiles'] = final_data['smiles'].apply(lambda x: map_mol(x))
    test_set = final_data.loc[final_data.smiles.isin(smiles)]
    train_set = final_data.loc[~final_data.smiles.isin(smiles)]

    final_data.to_pickle("../../data/dataset_surrogate_model_cs.pkl")
    test_set.to_pickle("../../data/testset_surrogate_model.pkl")
    train_set.to_pickle("../../data/trainset_surrogate_model.pkl")


def map_mol(smiles):

    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    [atom.SetAtomMapNum(idx + 1) for idx, atom in enumerate(mol.GetAtoms())]
    smiles = Chem.MolToSmiles(mol)

    return smiles


def info_1(rxn_data):

    frag = get_frag(rxn_data)
    frozen_dG = rxn_data.BDE_fr
    dG = (((DB.loc[DB.SMILES == frag].FreeEnergy.values + H_FreeEnergy) - (DB.loc[DB.SMILES == rxn_data.molecule].FreeEnergy.values)) * 627.509)
    spin_densities = literal_eval(DB.loc[DB.SMILES == frag].AtomSpins.values.tolist()[0])
    charges_all_atom = literal_eval(DB.loc[DB.SMILES == frag].AtomCharges.values.tolist()[0])

    return {
        "smiles": frag,
        "dG": [float(dG)],
        "frozen_dG": [frozen_dG],
        "spin_densities": spin_densities,
        "charges_all_atom": charges_all_atom
        }


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


def get_frag(serie) -> str:
    """ Return the fragment that is != [H] """

    return serie.fragment2 if serie.fragment2 != '[H]' else serie.fragment1


def multiprocess_func(func, elements, num_cpu=multiprocessing.cpu_count(), verbose=True) -> tuple:
    """ This function will run the func in multiples cpus """
    invalid_temp = 0
    outputs = []
    logging.info(f"Running {func} in {num_cpu} CPUs")
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
                logging.info(f"{func} call took more than {error.args} seconds")
                invalid_temp += 1
            except ValueError as error:
                logging.info(f"{func} failed due to ValueError: {error.args}")
                invalid_temp += 1
                raise
            except:
                logging.info(f"{func} failed due to an undefined error.")
                pass
        pool.close()
        pool.join()
        end_time = time.time()

        if verbose:
            with open("info.txt", "a") as file:
                file.write(f"\nThe {func} last {end_time - start_time} seconds")

    return (outputs, invalid_temp)


def get_number_surrogate():

    final_data = pd.read_pickle("../../data/dataset_surrogate_model_cs.pkl")
    final_data['dG'] = final_data['dG'].apply(lambda x: x[0])
    radicals = final_data.loc[final_data['dG'] != 0.0]
    molecules = final_data.loc[final_data['dG'] == 0.0]
    num_heavy_atoms_rad = 0
    num_heavy_atoms_mol = 0
    num_atoms_rad = 0
    num_atoms_mol = 0
    for row in final_data.itertuples():
        smiles = row.smiles
        mol = Chem.MolFromSmiles(smiles)
        if row.dG:
            num_heavy_atoms_rad += mol.GetNumAtoms()
            num_atoms_rad += len(row.charges_all_atom)
        else:
            num_heavy_atoms_mol += mol.GetNumAtoms()
            num_atoms_mol += len(row.charges_all_atom)





if __name__ == "__main__":
    create_train_test_surrogate_model()