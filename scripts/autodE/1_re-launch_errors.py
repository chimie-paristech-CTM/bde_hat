from shutil import rmtree
from glob import glob
import os

# rxns did not finish because the limit time, will be re-launched with more time
# rxns did not finish because some error, will be re-launched with more time


def new_inputs():

    # create input of the reactions that did not finish because of error
    autodE_errors()
    os.remove('0.autode_resume_dir/errors.txt')

    # create input of the reactions that did not finish because of time
    autodE_time()


def autodE_errors():

    rxns = get_errors('0.autode_resume_dir/errors.txt')

    for rxn in rxns:

        folder = f"rxn_{rxn}"

        rmtree(f"0.autode_resume_dir/{folder}")
        os.remove(f"{folder}/reaction.tar.gz")
        os.remove(f"{folder}/ade_{rxn}.log")

        out_files = glob(f"{folder}/ade_{rxn}_*")
        [os.remove(file) for file in out_files]

        with open('re-launch.sh', 'a') as sh_file:
            sh_file.write(f"cd {folder}\n")
            sh_file.write(f"sbatch slurm_{rxn}.sh \n")
            sh_file.write(f"cd ../\n")

    os.remove('0.autode_resume_dir/errors.txt')


def get_errors(file):

    with open(file, 'r') as err_file:
        lines = err_file.readlines()

    rxns = [line.split()[1] for line in lines]

    return rxns


def autodE_time():
    """ Delete the old files and create the new slurm"""

    pwd = os.getcwd()
    time_stop()

    with open('0.autode_resume_dir/time_stop.txt', 'r') as time_file:
        lines = time_file.readlines()

    folders = [line.split()[0] for line in lines[1:]]

    for folder in folders:

        os.chdir(folder)
        idx = folder[4:]

        rmtree('reaction')
        os.remove(f"slurm_{idx}.sh")
        os.remove(f"ade_{idx}.log")

        other_files = glob(f"ade_{idx}_*")

        if other_files:
            for file in other_files:
                os.remove(file)

        create_slurm(int(idx), 'autodE')

        os.chdir(pwd)

        with open('re-launch.sh', 'a') as sh_file:
            sh_file.write(f"cd {folder}\n")
            sh_file.write(f"sbatch slurm_{idx}.sh \n")
            sh_file.write(f"cd ../\n")


def time_stop():

    folders = glob('rxn_*')

    current_dir = os.getcwd()
    time_file = os.path.join(current_dir, '0.autode_resume_dir/time_stop.txt')

    with open(time_file, 'w') as file_ou:
        file_ou.write("Reactions that reached time limit\n")

    for folder in folders:

        os.chdir(folder)

        err_file = glob('*.err')

        if err_file:

            with open(err_file[0], 'r') as err:
                err_lines = err.readlines()

            if 'DUE TO TIME LIMIT' in err_lines[-1]:
                with open(time_file, 'a') as file_ou:
                    file_ou.write(f"{folder} \n")

        os.chdir('../')


def create_slurm(idx, conda_env='autodE'):
    """ Create the slurm input """

    # Setting the calculation

    nodes = 1
    tasks_per_node = 1
    cpus_per_task = 24
    log_level = 'INFO'  # {DEBUG, INFO, WARNING, ERROR}

    file_name = f"slurm_{idx:07}.sh"
    ade_idx = f"{idx:07}"

    with open(f"{file_name}", 'w') as in_slurm:
        in_slurm.write('#!/bin/bash\n')
        in_slurm.write(f"#SBATCH --job-name=ade_{ade_idx}\n")
        in_slurm.write(f"#SBATCH --nodes={nodes}\n")
        in_slurm.write(f"#SBATCH --ntasks-per-node={tasks_per_node}\n")
        in_slurm.write(f"#SBATCH --cpus-per-task={cpus_per_task}\n")
        in_slurm.write('#SBATCH --qos=qos_cpu-t4\n')
        in_slurm.write('#SBATCH --time=100:00:00\n')
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

    return None


if __name__ == "__main__":
    new_inputs()
