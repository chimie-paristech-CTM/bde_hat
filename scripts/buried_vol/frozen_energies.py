import pandas as pd
from rdkit import Chem
import logging
import time
from concurrent.futures import TimeoutError
from pebble import ProcessPool
from rdkit import RDLogger
import os
import multiprocessing
from glob import glob
import subprocess

RDLogger.DisableLog('rdApp.*')

hydrogen_rad_energy = -0.393482763936  # Hartree, calculated at GFN2 level


def main_function():

    paton_rxns = pd.read_csv('../../data/reactions_db_wo_duplicates.csv', index_col=0)
    radicals_1 = paton_rxns['fragment2'].tolist()
    radicals_2 = paton_rxns['fragment1'].tolist()
    radicals = radicals_2 + radicals_1
    radicals = list(set(radicals))
    del radicals[radicals.index('[H]')]

    global mol_cod_xyz
    mol_cod_xyz = pd.read_csv('../../data/df_smile_mol_code.csv', index_col=0)

    energy_mol = multiprocess_func(get_energy_molecule, mol_cod_xyz.iloc, 70)
    mol_cod_xyz['xtb_energy'] = energy_mol[0]
    mol_cod_xyz.to_csv('../../data/df_smile_mol_code_energy.csv')
    mol_cod_xyz = mol_cod_xyz.loc[mol_cod_xyz.xtb_energy != 'False']
    paton_rxns = paton_rxns.loc[paton_rxns['molecule'].isin(mol_cod_xyz.smiles)]

    if not os.path.exists('../../xyz_frozen_paton_rad'):
        os.mkdir('../../xyz_frozen_paton_rad')

    error = multiprocess_func(create_xyz_frozen_radical_parallel, paton_rxns.iloc, 70)
    os.chdir('../../xyz_frozen_paton_rad')

    xyz_frozen_files = glob('*.xyz')
    error = multiprocess_func(run_xtb, xyz_frozen_files, 15)
    os.chdir('../scripts/buried_vol')

    all_bdfe_frozen = multiprocess_func(get_energy_parallel, paton_rxns.iloc, 70)

    paton_rxns['BDFE_fr'] = all_bdfe_frozen[0]
    paton_rxns = paton_rxns.loc[paton_rxns['BDFE_fr'] < 1000]
    paton_rxns.to_csv('../../data/paton_rxns_frozen_BDFE.csv')



def run_xtb(file):

    xtb_path = '/home/javialra/soft/xtb-6.6.0/bin'
    xtb_command = os.path.join(xtb_path, 'xtb')

    out_file = f"{file[:-4]}.log"

    with open(out_file, 'w') as out:
        subprocess.run(f"{xtb_command} {file} --cycles 1000 --chrg 0 -P 4 --uhf 1", shell=True, stdout=out, stderr=out)

    subprocess.run('rm xtbrestart charges wbo xtbtopo.mol', shell=True)

    return None


def get_energy_molecule(data):

    xyz_file = f"../../xtb_opt_paton/{data.xyz_file}.log"
    with open(xyz_file, 'r') as file:
        lines = file.readlines()[::-1]
    for line in lines:
        if '[ERROR]' in line:
            return False
        if 'TOTAL ENERGY' in line:
            energy = float(line.split()[3])
        if line.startswith('          :  # imaginary freq.'):
            if line.split()[4] != '0':
                return False

    return energy


def get_energy_parallel(data):
    mol = data.molecule
    rad_file = f"rxn_{data.name}_frozen.log"
    mol_file = f'{mol_cod_xyz.loc[mol_cod_xyz.smiles == mol].xyz_file.values[0]}.log'
    mol_file = os.path.join('../../xtb_opt_paton', mol_file)
    rad_file = os.path.join('../../xyz_frozen_paton_rad', rad_file)
    mol_energy = read_log(mol_file)
    rad_energy = read_log(rad_file)

    bdfe_fr = ((rad_energy + hydrogen_rad_energy) - mol_energy) * 627.509
    #output = (data.name, bdfe_fr)

    return bdfe_fr


def read_log(path):
    with open(path, 'r') as file:
        lines = file.readlines()[::-1]
        for line in lines:
            if 'TOTAL ENERGY' in line:
                return float(line.split()[3])
            if '[ERROR]' in line:
                return 1000
    return 1000


def create_xyz_frozen_radical_parallel(data):
    """ For the calculation of BDFE frozen, we need the geometry of the relaxed molecule and delete the H abstracted"""

    mol = Chem.MolFromSmiles(data.molecule)
    mol = Chem.AddHs(mol)

    bond = mol.GetBondWithIdx(int(data.bond_index))
    idx_H = bond.GetEndAtomIdx()

    xyz_file = mol_cod_xyz.loc[mol_cod_xyz.smiles == data.molecule].xyz_file.values[0]
    xyz_file_path = f"../../xtb_opt_paton/{xyz_file}_opt.xyz"
    path = os.getcwd()

    with open(xyz_file_path, 'r') as xyz_file:
        lines = xyz_file.readlines()

    del lines[idx_H + 2]  # + 2 because the first line is amount of atoms and second line is a comment

    _, name = os.path.split(xyz_file_path)
    name = f"rxn_{data.name}_frozen.xyz"

    final_path = os.path.join(path, '../../xyz_frozen_paton_rad')
    xyz_frozen = os.path.join(final_path, name)

    num_atoms = str(len(lines) - 2) + "\n"
    lines[0] = num_atoms

    with open(xyz_frozen, 'a') as xyz_frozen_file:
        for line in lines:
            xyz_frozen_file.write(line)

    return None


def get_rad_index(smiles):
    """ Get the index of the radical atom"""

    mol = Chem.MolFromSmiles(smiles)

    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            idx = atom.GetIdx()

    return idx


def get_code_from_xyz_parallel(xyz_file):
    with open(xyz_file, 'r') as f:
        lines = f.readlines()

    smiles = lines[1].split()[1]

    return (smiles, xyz_file)


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