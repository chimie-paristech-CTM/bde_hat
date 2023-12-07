#!/usr/bin/python
import os
import subprocess
import pandas as pd
from glob import glob
from xyz2mol import read_xyz_file, xyz2mol
from rdkit import Chem
import shutil


def post_processing_step():

    """    
    The idea behind all these functions are:

        Before all the calculations, we know which rxns should be checked, because the TS could lead to the wrong stereo-product

        with check_TS_stereochemistry(), we create the input for the optimization with xtb.

        with run_xtb, we launch the xtb optimization and check if it finishes normally

        with xyz_TS_to_mol, we get the rdkit molecule of the xtb optimization and that molecule will be the input for

        is_the_same_stereochemistry, this function compares that all the stereocenters matched between the product and the TS

        if True, nothing more to do but if it is False, we have to change the stereochemistry of the prod and re-run autodE
    """

    filter_freq()
    filter_connectivity()
    check_TS_stereochemistry()
    create_new_rxn()
    create_final_db_reactivity()


def filter_freq():

    df = pd.read_csv('../../data/autode_results.csv', index_col=0)
    df = df.loc[df.dG_act > 0]
    df = df.loc[df.frequency < -500]

    df_checked = pd.read_csv('../../data/rxns_freq_500.csv', index_col=0, sep=';')
    df_checked = df_checked.loc[df_checked['Check Freq']]
    df_checked.drop(columns=['Check Freq'], axis=1, inplace=True)

    df_final = pd.concat([df, df_checked], ignore_index=True)
    df_final.to_csv('../../data/autode_results_clean_freq.csv')


def filter_connectivity():

    df = pd.read_csv('../../data/autode_results_clean_freq.csv', index_col=0)
    df_checked = pd.read_csv('../../data/df_conn_check.csv', index_col=0, sep=';')
    df_checked = df_checked.loc[~df_checked['Check Conn']]
    df_final = df.loc[~df['ID'].isin(df_checked['ID'])]
    df_final.to_csv('../../data/autode_results_clean.csv')

def check_TS_stereochemistry() -> None:

    pwd = os.getcwd()
    outputs_ade = '../../autodE_input/0.autode_resume_dir'

    data_folder = os.path.join(pwd, '../../data')
    outputs_dir = os.path.join(pwd, outputs_ade)

    rxns_2k = pd.read_csv(f'{data_folder}/reactions_2k_autodE.csv', index_col=0)
    df = pd.read_csv(f'{data_folder}/autode_results_clean.csv', index_col=0)
    
    # the file check_TS_stereochemistry.txt was created in function stereochemistry_rxn in create_dataset.py

    with open(f'{data_folder}/check_TS_stereochemistry.txt', 'r') as file:
        rxns = [int(line.split()[0]) for line in file]

    rxns_check = rxns_2k.loc[rxns]
    rxns_check = rxns_check.loc[rxns_check.index.isin(df['ID'])]

    for rxn in rxns_check.itertuples():
        os.chdir(outputs_dir)
        H_idx = get_idx_H(rxn.reactions, rxn.bond_index_broken)
        atoms_r0 = get_num_atoms(rxn.reactions)
        r0, r1, p0, p1 = reacs_prods(rxn.reactions)
        p1_mol = obtain_molecule(p1)
        extract_geom(rxn.Index, H_idx, atoms_r0)
        run_xtb(rxn.Index)
        out_file = f"ts_check_{rxn.Index:07}_opt.out"
        out_xyz_file = f"ts_check_{rxn.Index:07}_opt.xyz"
        normal_termination = check_output_xtb(out_file)
        if normal_termination:
            copy_xyz_output(out_xyz_file)
        else:
            with open(f'{data_folder}/info_TS_stereochemistry.txt', 'a') as err_file:
                err_file.write(f"Error during optimization of rxn {rxn.Index:07}\n")
            continue
        ts_mol = xyz_TS_to_mol(out_xyz_file)
        if is_the_same_stereochemistry(ts_mol, p1_mol):
            with open(f'{data_folder}/info_TS_stereochemistry.txt', 'a') as info_file:
                info_file.write(f"TS of rxn {rxn.Index:07} is leading to the correct stereo-product\n")
        else:
            with open(f'{data_folder}/info_TS_stereochemistry.txt', 'a') as info_file:
                info_file.write(f"TS of rxn {rxn.Index:07} is leading to the wrong stereo-product\n")
            ts_mol_noHs = Chem.RemoveHs(ts_mol)
            ts_smiles = Chem.MolToSmiles(ts_mol_noHs)
            ts_smiles = canonicalize_smiles(ts_smiles)
            optimize_new_product(ts_smiles, rxn.Index)

    os.chdir(pwd)

    return None


def get_idx_H(smiles : str, bond_index_broken : int) -> int:

    r0 = reacs_prods(smiles)[0]
    mol = obtain_molecule(r0)
    bond_broken  = mol.GetBondWithIdx(bond_index_broken) 

    return bond_broken.GetEndAtomIdx() if bond_broken.GetEndAtom().GetSymbol() == 'H' else bond_broken.GetBeginAtomIdx() 


def get_num_atoms(smiles : str) -> int:

    r0 = reacs_prods(smiles)[0]
    mol = obtain_molecule(r0)

    return mol.GetNumAtoms()


def extract_geom(rxn_idx : int, H_idx : int, atoms_r0 : int) -> None:

    os.chdir(f"rxn_{rxn_idx:07}")

    ts_file = f"TS_{rxn_idx:07}.xyz"
    
    geom = []
    with open(ts_file, 'r') as file:
        geom = file.readlines()

    del geom[:2]

    H_r0 = geom[H_idx]
    r1 = geom[atoms_r0:]

    r1.append(H_r0)

    os.mkdir('ts_check')
    xyz_file = f"ts_check_{rxn_idx:07}.xyz"
    
    with open(f"ts_check/{xyz_file}", 'w') as new_file:
        new_file.write(f"{len(r1)}\n")
        new_file.write("Generated for bde_project to check the 'stereochemistry' of the TS\n")
        for atom in r1:
            new_file.write(atom)

    return None


def run_xtb(rxn_idx : int) -> None:
    
    os.chdir('ts_check')

    xtb_path = '/home/javialra/soft/xtb-6.6.0/bin'
    xtb_command = os.path.join(xtb_path, 'xtb')

    in_xyz_file = f"ts_check_{rxn_idx:07}.xyz"
    out_file = f"ts_check_{rxn_idx:07}_opt.out"
    command_line = f"{xtb_command} {in_xyz_file} --opt --parallel 4"

    with open(out_file, 'w') as out:
        subprocess.run(f"{xtb_command} {in_xyz_file} --opt --parallel 4", shell=True, stdout=out, stderr=out)

    return None


def check_output_xtb(out_file : str) -> bool:

    out_lines = []

    with open(out_file, 'r') as file:
        out_lines = file.readlines() 

    opt_criteria = "   *** GEOMETRY OPTIMIZATION CONVERGED AFTER "
    
    for line in reversed(out_lines):
        if "ERROR" in line:
            return False
        if opt_criteria in line:
            return True

    return False


def copy_xyz_output(xyz_file : str) -> None:
    
    out_xyz_xtb = 'xtbopt.xyz'
    os.rename(out_xyz_xtb, xyz_file)

    return None


def xyz_TS_to_mol(xyz_file : str) -> object:
    """ Convert 3D coordinates into a rkdit molecule,
    Using xyz2mol, from https://github.com/jensengroup/xyz2mol
    """

    atoms, charge, xyz_coordinates = read_xyz_file(xyz_file)
    mol = xyz2mol(atoms, xyz_coordinates, charge)[0]

    return mol


def is_the_same_stereochemistry(mol_TS : object, mol_prod : object) -> bool:
    """ Compare if the TS is leading to the correct stereo_product """

    stereo_info_TS = Chem.FindMolChiralCenters(mol_TS, force=True, includeUnassigned=True, useLegacyImplementation=True)

    stereo_info_prod = Chem.FindMolChiralCenters(mol_prod, force=True, includeUnassigned=True, useLegacyImplementation=True)

    return True if stereo_info_prod == stereo_info_TS else False


def canonicalize_smiles(smiles):
    """ Return a consistent SMILES representation for the given molecule """
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol)


def optimize_new_product(smiles : str, rxn_idx : int) -> None:

    os.chdir('../')
    os.makedirs(f"opt_alt_p1_{rxn_idx:07}")
    working_dir = os.path.join(os.getcwd(), f"opt_alt_p1_{rxn_idx:07}")
    os.chdir(working_dir)
    input_ade(smiles, rxn_idx)
    input_bash(rxn_idx)
    run_ade(rxn_idx)

    return None


def input_ade(smiles : str, rxn_idx : int) -> None:
    """ Create the input for the new ade calculation """

    with open(f"p1_alt_{rxn_idx:07}.py", 'w') as ade_input:
        ade_input.write('import autode as ade\n')
        ade_input.write('from autode.wrappers.G16 import g16\n')
        ade_input.write('import time\nt0=time.time()\n\n')
        ade_input.write('def print_energies_to_csv(mol):\n')
        ade_input.write('\twith open(\'energies.csv\', \'w\') as f:\n')
        ade_input.write('\t\tf.write(\'Species, E_opt, G_cont, H_cont, E_sp\\n\')\n')  #revisar
        ade_input.write("\t\tf.write(f'{mol.name}, {mol.energies.first_potential}, {mol.g_cont}, {mol.h_cont}, {mol.energies.last_potential}')\n\n")
        ade_input.write('ade.Config.n_cores=40\n')
        ade_input.write('ade.Config.max_core=4000\n')
        ade_input.write('g16.keywords.set_functional(\'um062x\')\n')
        ade_input.write('g16.keywords.set_dispersion(\'None\')\n')
        ade_input.write(f"kwds_sp = ade.Config.G16.keywords.sp\n")
        ade_input.write(f"kwds_sp.append(' stable=opt')\n")
        ade_input.write(f"ade.Config.num_conformers=1000\n")
        ade_input.write(f"ade.Config.rmsd_threshold=0.1\n")
        ade_input.write(f"product = ade.Molecule(name='p1_{rxn_idx:07}', smiles=r\"{smiles}\")\n")
        ade_input.write('product.find_lowest_energy_conformer(hmethod=g16)\n')
        ade_input.write('product.optimise(method=g16)\n')
        ade_input.write('product.calc_thermo(method=g16)\n')
        ade_input.write('product.single_point(method=g16)\n')
        ade_input.write('product.print_xyz_file()\n')
        ade_input.write('print_energies_to_csv(product)\n')
        ade_input.write('t1=time.time()\nprint(f"Duration: {t1-t0}")\n')

    return None


def input_bash(rxn_idx : int) -> None:
    """ Create the input for the sh input """

    with open(f"p1_alt_{rxn_idx:07}.sh", 'w') as bash_input:
        bash_input.write('__conda_setup="$(\'/home/javialra/anaconda3/bin/conda\' \'shell.bash\' \'hook\' 2> /dev/null)"\n')
        bash_input.write('if [ $? -eq 0 ]; then\n')
        bash_input.write('\teval "$__conda_setup"\n')
        bash_input.write('else\n')
        bash_input.write('\tif [ -f "/home/javialra/anaconda3/etc/profile.d/conda.sh" ]; then\n')
        bash_input.write('\t\t. "/home/javialra/anaconda3/etc/profile.d/conda.sh"\n')
        bash_input.write('\telse\n')
        bash_input.write('\t\texport PATH="/home/javialra/anaconda3/bin:$PATH"\n')
        bash_input.write('\tfi\n')
        bash_input.write('fi\n')
        bash_input.write('conda activate autoDE_env\n')
        bash_input.write(f"python3 p1_alt_{rxn_idx:07}.py\n")

    return None


def run_ade(rxn_idx : int):
    """ Run the ade calculation"""

    chmod = f"chmod +x p1_alt_{rxn_idx:07}.sh"
    subprocess.run(chmod, shell=True)

    ade_command = f"./p1_alt_{rxn_idx:07}.sh"
    out_file = f"./p1_alt_{rxn_idx:07}.out"

    with open(out_file, 'w') as out:
        subprocess.run(ade_command, shell=True, stdout=out, stderr=out)


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


def obtain_molecule(smiles: str, addH = True) -> object:
    """ Return a molecule for the given SMILES """

    molecule = Chem.MolFromSmiles(smiles)
    if addH:
        molecule = Chem.rdmolops.AddHs(molecule)
    Chem.Kekulize(molecule, clearAromaticFlags=True)

    return molecule


def create_new_rxn():
    with open('../../data/info_TS_stereochemistry.txt', 'r') as info_file:
        info = info_file.readlines()

    wrong_rxns = [line.split()[3] for line in info if 'wrong' in line]

    current_dir = os.getcwd()
    calculation_dir = '../../autodE_input/0.autode_resume_dir/'
    os.makedirs('../../autodE_input/0.autode_resume_dir/corrected_rxns')

    corrected_info = []

    for rxn in wrong_rxns:

        rel_path_final_dir = f"../../autodE_input/0.autode_resume_dir/corrected_rxns/rxn_{rxn}"
        rel_freq = f"../../autodE_input/0.autode_resume_dir/corrected_rxns/rxn_{rxn}/frequency_logs"
        rel_sp = f"../../autodE_input/0.autode_resume_dir/corrected_rxns/rxn_{rxn}/single_points_logs"

        os.chdir(current_dir)
        os.chdir(f"{calculation_dir}/rxn_{rxn}")

        if not check_calc_finished(rxn):
            with open('../../data/errors_post_processing.txt', 'a') as err_file:
                err_file.write(f"{rxn}\n")
            continue

        final_dir = create_dir(rel_path_final_dir, current_dir)
        freq_dir = create_dir(rel_freq, current_dir)
        sp_dir = create_dir(rel_sp, current_dir)

        copy_xyz_files(rxn, final_dir)
        copy_sp_files(rxn, sp_dir)
        copy_freq_files(rxn, freq_dir)

        with open('rxn_smile.txt', 'r') as old_rxn_file:
            rxn_smiles = old_rxn_file.readline()

        copy_new_prod(rxn, final_dir)

        with open(f"p1_alt_{rxn}.py", 'r') as new_file:
            for line in new_file:
                if line.startswith('product = ade.Molecule'):
                    new_prod_smiles = line.split()[3][9:-2]
                    break

        os.chdir('../')

        new_rxn_smiles = f"{rxn_smiles.split('.')[0]}.{rxn_smiles.split('.')[1]}.{new_prod_smiles}"

        with open(f"{final_dir}/rxn_smile.txt", 'a') as new_rxn_file:
            new_rxn_file.write(new_rxn_smiles)

        old_energies = pd.read_csv('energies.csv', index_col=0)

        energy_p1 = pd.read_csv(f"opt_alt_p1_{rxn}/energies.csv", index_col=0)

        energy_p1.rename(index={f'{energy_p1.index[0]}': f'{energy_p1.index[0]}_alt'}, inplace=True)

        final_energies = pd.concat([old_energies, energy_p1])

        final_energies.to_csv(f"{final_dir}/energies.csv")

        dG_act, dG_rxn = extract_energy(f"{final_dir}/energies.csv")

        rxn_info_corrected = (rxn, new_rxn_smiles, dG_act, dG_rxn)

        corrected_info.append(rxn_info_corrected)

    os.chdir(current_dir)
    ade_results_corrected = pd.DataFrame(corrected_info, columns=['ID', 'RXN_SMILES', 'dG_act', 'dG_rxn'])
    ade_results_corrected.to_csv('../../data/autode_results_new_rxns_ts.csv')


def extract_energy(energies_file):
    """ From energy.csv return Free Gibbs Energy of Activation and of Reaction"""

    prods = []
    reacs = []

    energies = pd.read_csv(energies_file)

    for specie in energies.Species:

        if specie.startswith('r'):
            reacs.append(energies.loc[energies.Species == specie].index[0])
        elif specie.startswith('p0') or specie.endswith('_alt'):
            prods.append(energies.loc[energies.Species == specie].index[0])
        elif specie.startswith('TS'):
            ts = energies.loc[energies.Species == specie].index[0]

    G_prods = 0
    G_reacs = 0

    for prod in prods:
        G_prods += (energies[" E_sp"][prod] + energies[" G_cont"][prod])

    for reac in reacs:
        G_reacs += (energies[" E_sp"][reac] + energies[" G_cont"][reac])

    G_ts = energies[" E_sp"][ts] + energies[" G_cont"][ts]

    dG_rxn = (G_prods - G_reacs) * 627.509
    dG_act = (G_ts - G_reacs) * 627.509

    return dG_act, dG_rxn


def create_dir(relative_path: str, pwd: str):
    """ Create a dir """

    absolute_path = os.path.join(pwd, relative_path)

    if not os.path.isdir(absolute_path):
        os.makedirs(absolute_path)

    return absolute_path


def copy_xyz_files(idx: str, dst_dir: str) -> None:
    """ Copy xyz files of TS, prod and reacs """

    prods = glob("p*")
    tss = glob("TS*")
    reacs = glob("r*xyz")

    [shutil.copy(f"{prod}", f"{dst_dir}/{prod}") for prod in prods]
    [shutil.copy(f"{reac}", f"{dst_dir}/{reac}") for reac in reacs]
    [shutil.copy(f"{ts}", f"{dst_dir}/{ts}") for ts in tss]

    return None


def copy_sp_files(idx: str, dst_dir: str) -> None:
    """ Copy the sp outputs of autode to a new directory"""

    sp_dir = 'single_points_logs'
    os.chdir(sp_dir)

    log_files = glob('*.log')

    [shutil.copy(f"{file}", f"{dst_dir}/{file}") for file in log_files]

    os.chdir('../')

    return None


def copy_freq_files(idx: str, dst_dir: str) -> None:
    """ Copy the hessian outputs of autode to a new directory"""

    freq_dir = 'frequency_logs'
    os.chdir(freq_dir)

    log_files = glob('*.log')

    [shutil.copy(f"{file}", f"{dst_dir}/{file}") for file in log_files]

    os.chdir('../')

    return None


def copy_new_prod(idx, dst_dir) -> None:
    os.chdir(f"opt_alt_p1_{idx}")

    log_files = glob('*log')

    for log in log_files:
        if 'sp' in log:
            shutil.copy(f"{log}", f"{dst_dir}/single_points_logs/alt_{log}")
        elif 'hess' in log:
            shutil.copy(f"{log}", f"{dst_dir}/frequency_logs/alt_{log}")

    shutil.copy(f"p1_{idx}.xyz", f"{dst_dir}/alt_p1_{idx}.xyz")

    return None


def check_calc_finished(idx) -> bool:
    with open(f"opt_alt_p1_{idx}/p1_alt_{idx}.out", 'r') as ou_file:
        lines = ou_file.readlines()

    return False if 'Duration' not in lines[-1] else True


def create_final_db_reactivity():
    """
        delete rxns that give the wrong products
    """

    db_reactivity = pd.read_csv('../../data/autode_results_clean.csv', index_col=0)
    db_reactivity.rename(columns={'ID': 'rxn_id'}, inplace=True)
    db_reactivity.drop(columns=['frequency'], inplace=True)
    db_reactivity.set_index('rxn_id', inplace=True)
    db_corrected_rxns = pd.read_csv('../../data/autode_results_new_rxns_ts.csv', index_col=0)
    db_corrected_rxns.rename(columns={'ID': 'rxn_id'}, inplace=True)

    rxns_error = []
    rxns_drop = db_corrected_rxns.rxn_id.tolist()

    if os.path.isfile('../../data/errors_post_processing.txt'):
        with open('../../data/errors_post_processing.txt', 'r') as er_file:
            for line in er_file:
                rxns_error.append(int(line[:-1]))
        rxns_drop += rxns_error

    final_db_reactivity = db_reactivity.drop(rxns_drop)

    db_corrected_rxns.rxn_id = db_corrected_rxns.rxn_id + 2000000  # just to identify later which rxns are different from the original one
    db_corrected_rxns.set_index('rxn_id', inplace=True)

    final_db_reactivity = pd.concat([final_db_reactivity, db_corrected_rxns], ignore_index=False)
    final_db_reactivity.drop(columns=['time'], inplace=True)

    final_db_reactivity.to_csv('../../data/reactivity_database.csv')

    return None


if __name__ == "__main__":
    post_processing_step()
