import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import NumRadicalElectrons
from morfeus import read_xyz, BuriedVolume
import logging
import time
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from rdkit import RDLogger
import os
import multiprocessing
RDLogger.DisableLog('rdApp.*')


def main_function():

    db_paton = pd.read_csv('../../data/20200415_radical_database.csv', index_col=0)
    paton_rxns = pd.read_csv('../../data/reactions_db_wo_duplicates.csv', index_col=0)

    radicals_1 = paton_rxns['fragment2'].tolist()
    radicals_2 = paton_rxns['fragment1'].tolist()

    radicals = radicals_2 + radicals_1
    radicals = list(set(radicals))

    molecules = list(set(paton_rxns['molecule'].tolist()))

    del radicals[radicals.index('[H]')]

    df_code = extract_xyz_files()

    input_bv_parallel = []

    mol_code = []

    for mol in molecules:
        mol_code.append((mol, df_code.loc[df_code['smiles'] == mol].code_xyz.values[0]))

    global mol_cod_xyz
    mol_cod_xyz = pd.DataFrame(mol_code, columns=['smiles', 'xyz_file'])
    mol_cod_xyz.to_csv('../../data/df_smile_mol_code.csv')

    for rad in radicals:
        input_bv_parallel.append((rad, df_code.loc[df_code['smiles'] == rad].code_xyz.values[0]))

    df_input_bv_parallel = pd.DataFrame(input_bv_parallel, columns=['smiles', 'xyz_file'])

    bv_output = multiprocess_func(get_buried_vol_parallel, df_input_bv_parallel.iloc, 70)
    df_input_bv_parallel['Buried_Vol'] = bv_output[0]
    df_input_bv_parallel.to_csv('../../data/df_buried_vol.csv')


def extract_xyz_files():

    mol_suppl = Chem.SDMolSupplier('../../data/20200415_radical_database.sdf', removeHs=False)

    if not os.path.exists('../../xyz_paton'):
        os.mkdir('../../xyz_paton')

    number = 1

    smiles_name = []

    for mol in mol_suppl:
        xyz = mol.GetConformer().GetPositions()
        symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
        smiles = mol.GetPropsAsDict()['SMILES']
        name = 'molecule' if NumRadicalElectrons(mol) == 0 else 'fragment'
        num_atoms = mol.GetNumAtoms()
        with open(f"../../xyz_paton/{name}_{number}.xyz", 'a') as xyz_file:
            xyz_file.write(f"{num_atoms}\n")
            xyz_file.write(f"SMILES: {smiles}\n")
            for coord, symbol in zip(xyz, symbols):
                xyz_file.write(f"{symbol}  {coord[0]}  {coord[1]}  {coord[2]}\n")
        smiles_name.append((f"{smiles}", f"{name}_{number}"))
        number += 1

    df_code = pd.DataFrame(smiles_name, columns=['smiles', 'code_xyz'])
    df_code.to_csv('../../data/df_code_smiles_xyz.csv')

    return df_code


def get_buried_vol_parallel(data):

    """ Get buried vol
    Args:
    """

    xyz_file = data['xyz_file']
    mol = Chem.MolFromSmiles(data['smiles'])
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            idx = atom.GetIdx()
            break
    elements, coordinates = read_xyz(f'../../xyz_paton/{xyz_file}.xyz')
    bv = BuriedVolume(elements, coordinates, idx + 1)

    return bv.fraction_buried_volume


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


if __name__ == "__main__":
    main_function()

