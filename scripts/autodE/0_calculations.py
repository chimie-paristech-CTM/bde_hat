#!/usr/bin/python

import os
import shutil
import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--csv_file', type=str, default='../../data/reactions_2k_autodE.csv',
                    help='path to csv file containing reactions')
parser.add_argument('--final_dir', type=str, default='../../autodE_input',
                    help='path to the folder with the autodE input')
parser.add_argument('--conda_env', type=str, default='autodE',
                    help='conda environment of autodE package')


def create_input_autodE(final_dir, csv_file, conda_env="autodE"):

    create_input(csv_file, final_dir, conda_env)

    return None


def create_input_g16_xtb():

    dir_g16 = 'ranking_g16'
    dir_xtb = 'ranking_xtb'

    create_input('../../data/subset_30_g16_xtb_autodE.csv', dir_g16, 'autodE')
    create_input('../../data/subset_30_g16_xtb_autodE.csv', dir_xtb, 'autodE', hmet_confor=r"False")

    return None


def create_input(csv_file, final_dir, conda_env, hmet_confor=r"True"):
    """ Create the input for autoDE"""
    
    current_dir = os.getcwd()
    aux_script = 'aux_script.py'
    aux_script_dir = os.path.join(current_dir, aux_script)

    rxns = pd.read_csv(csv_file, index_col=0)

    # checking directory
    if os.path.isdir(final_dir):
        os.chdir(final_dir)
    else:
        os.mkdir(final_dir)
        os.chdir(final_dir)

    for rxn in rxns.itertuples():

        rxn_smile = rxn.rxn_smiles
        idx = rxn.rxn_id

        directory = f"rxn_{idx:07}"
        os.mkdir(directory)

        create_ade_input(rxn_smile, idx, directory, hmet_confor=hmet_confor)
        create_slurm(idx, directory, conda_env)
        shutil.copy(aux_script_dir, directory)

        with open(f"{directory}/rxn_smile.txt", 'w') as rxn_file:
            rxn_file.write(rxn_smile)
    
    os.chdir(current_dir)

    return None


def create_ade_input(rxn_smile, idx, dir, hmet_confor=r"True"):
    """ Create the ade input """

    # Setting the calculation

    functional = 'um062x'
    cores = 24
    mem = 4000
    num_conf = 1000
    rmsd = 0.1
    dispersion = r"None"

    file_name = f"ade_{idx:07}.py"

    with open(f"{dir}/{file_name}", 'w') as in_ade:
        in_ade.write('import autode as ade\n')
        in_ade.write('import time\n')
        in_ade.write('t0 = time.time()\n')
        in_ade.write(f"ade.Config.n_cores={cores}\n")
        in_ade.write(f"ade.Config.max_core={mem}\n")
        in_ade.write(f"ade.Config.hcode=\"G16\"\n")
        in_ade.write(f"ade.Config.lcode =\"xtb\"\n")
        in_ade.write(f"rxn=ade.Reaction(r\"{rxn_smile}\")\n")
        #in_ade.write(f"rxn=ade.Reaction(r\"{rxn_smile}\", solvent_name = 'water')\n")
        in_ade.write(f"ade.Config.G16.keywords.set_functional('{functional}')\n")
        in_ade.write(f"ade.Config.G16.keywords.set_dispersion({dispersion})\n")
        in_ade.write(f"kwds_sp = ade.Config.G16.keywords.sp\n")
        in_ade.write(f"kwds_sp.append(' stable=opt')\n")
        #in_ade.write(f"kwds_xtb_low_opt = ade.Config.XTB.keywords.low_opt\n")
        #in_ade.write(f"kwds_xtb_low_opt.append('--opt')\n")
        #in_ade.write(f"kwds_xtb_low_opt.append('--iterations 750')\n")
        in_ade.write(f"ade.Config.num_conformers={num_conf}\n")
        in_ade.write(f"ade.Config.rmsd_threshold={rmsd}\n")
        in_ade.write(f"ade.Config.hmethod_conformers={hmet_confor}\n")
        in_ade.write('rxn.calculate_reaction_profile(free_energy=True)\n')
        in_ade.write('t1 = time.time()\n')
        in_ade.write('print(f"Duration: {t1-t0}")\n')

    return None


def create_slurm(idx, dir, conda_env):
    """ Create the slurm input """

    # Setting the calculation

    nodes = 1
    tasks_per_node = 1
    cpus_per_task = 24
    log_level = 'INFO' # {DEBUG, INFO, WARNING, ERROR}

    file_name = f"slurm_{idx:07}.sh"
    ade_idx = f"{idx:07}"

    with open(f"{dir}/{file_name}", 'w') as in_slurm:
        in_slurm.write('#!/bin/bash\n')
        in_slurm.write(f"#SBATCH --job-name=ade_{ade_idx}\n")   
        in_slurm.write(f"#SBATCH --nodes={nodes}\n")    
        in_slurm.write(f"#SBATCH --ntasks-per-node={tasks_per_node}\n")
        in_slurm.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")  
        in_slurm.write('#SBATCH --qos=qos_cpu-t3\n')
        in_slurm.write('#SBATCH --time=20:00:00\n')
        in_slurm.write('#SBATCH --hint=nomultithread  # Disable hyperthreading\n')
        in_slurm.write(f"#SBATCH --output=ade_{ade_idx}_%j.out\n")   
        in_slurm.write(f"#SBATCH --error=ade_{ade_idx}_%j.err\n") 
        in_slurm.write(f"#SBATCH --account=qev@cpu\n")
        in_slurm.write('module purge\n')
        in_slurm.write('module load xtb/6.4.1\n') 
        in_slurm.write('module load gaussian/g16-revC01\n')
        in_slurm.write('module load python\n') 
        in_slurm.write('export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK\n')
        in_slurm.write('export GAUSS_SCRDIR=$JOBSCRATCH\n')
        in_slurm.write(f"conda activate {conda_env}\n")
        in_slurm.write(f"export AUTODE_LOG_LEVEL={log_level}\n")
        in_slurm.write(f"export AUTODE_LOG_FILE=ade_{ade_idx}.log\n")
        in_slurm.write(f"python3 ade_{ade_idx}.py \n")
        in_slurm.write(f"python3 aux_script.py \n")
        #in_slurm.write(f"python3 aux_script_rmechdb.py \n")

    return None


if __name__ == "__main__":
    # set up
    args = parser.parse_args()
    create_input_autodE(args.final_dir, args.csv_file, args.conda_env)
