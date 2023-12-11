#!/usr/bin/python

import os
from glob import glob
import shutil
import tarfile

def copy_imp_files() -> None:

    idx_rxn = os.path.split(os.getcwd())[1][4:]

    pwd = os.getcwd()
    rel_path_final_dir = f"../0.autode_resume_dir/rxn_{idx_rxn}"
    rel_freq = f"../0.autode_resume_dir/rxn_{idx_rxn}/frequency_logs"
    rel_sp =f"../0.autode_resume_dir/rxn_{idx_rxn}/single_points_logs"

    create_dir(rel_path_final_dir, pwd)
    create_dir(rel_freq, pwd)
    create_dir(rel_sp, pwd)

    abs_path_final_dir = os.path.join(pwd, rel_path_final_dir)

    if rxn_completed(idx_rxn):

        shutil.copy('rxn_smile.txt', abs_path_final_dir)
        shutil.copy(glob("*.out")[0], abs_path_final_dir)
        shutil.copy(glob("*.err")[0], abs_path_final_dir)

        copy_files(idx_rxn, abs_path_final_dir)
        os.chdir(pwd)

    with tarfile.open('reaction.tar.gz', "w:gz") as tar:
        tar.add('reaction')
        
    shutil.rmtree('reaction')

    return None

def create_dir(relative_path : str, pwd : str) -> None:
    """ Create a dir """

    absolute_path = os.path.join(pwd, relative_path)

    if not os.path.isdir(absolute_path):
        os.makedirs(absolute_path)
    
    return None

def rxn_completed(idx : str) -> bool:
    """ Check if the autodE finished succesfully and if it finds the TS """

    pwd = os.getcwd()
    out_dir = os.path.join(pwd, 'reaction/output')

    if os.path.isdir(out_dir):
        if check_TS(out_dir):
            return True 
        else:
            mssg = f"rxn {idx} finished but did not find TS"
    else:
        mssg = f"rxn {idx} did not finished"

    error_file = f"../0.autode_resume_dir/errors.txt"
    with open(error_file, 'a') as err_file:
        err_file.write(f"{mssg}\n")

def check_TS(path) -> bool:
    """ Check if autodE finds a TS """

    files = glob(f"{path}/TS*")

    return True if files else False

def copy_files(idx : str, dst_dir : str) -> None:
    """ Copy the relevant outputs of autode to a new directory"""

    os.chdir('reaction/output')

    shutil.copy('energies.csv', f"{dst_dir}/")
    
    copy_xyz_files(idx, dst_dir)

    copy_sp_files(idx, dst_dir)

    copy_freq_files(idx, dst_dir)

    return None

def copy_xyz_files(idx : str, dst_dir : str) -> None:
    """ Copy xyz files of TS, prod and reacs """

    prods = glob("p*")
    tss = glob("TS*")
    reacs = glob("r*")
    reac_complex = glob('*_reactant.xyz')
    prod_complex = glob('*_product.xyz')

    [shutil.copy(f"{prod}", f"{dst_dir}/{prod[:2]}_{idx}.xyz") for prod in prods]
    [shutil.copy(f"{reac}", f"{dst_dir}/{reac[:2]}_{idx}.xyz") for reac in reacs]
    for ts in tss:
        if 'imag_mode' in ts:
            shutil.copy(f"{ts}", f"{dst_dir}/{ts[:-4]}_{idx}.xyz")
        else:
            shutil.copy(f"{ts}", f"{dst_dir}/{ts[:2]}_{idx}.xyz")
            ts_name = ts
    
    if reac_complex:
        shutil.copy(f"{reac_complex[0]}", f"{dst_dir}/{reac_complex[0].split('_')[1][:-4]}_complex_{idx}.xyz")

    if prod_complex:
        shutil.copy(f"{prod_complex[0]}", f"{dst_dir}/{prod_complex[0].split('_')[1][:-4]}_complex_{idx}.xyz")

    copy_ts_file(idx, dst_dir, ts_name)

    return None
    
def copy_ts_file(idx : str, dst_dir : str, ts_name : str) -> None:
    """ Copy the ts opt/freq output of autode to a new directory"""

    with open(ts_name, 'r') as ts_xyz:
        comment_line = ts_xyz.readlines()[1]

    freq_xyz = float(comment_line.split()[-2])

    ts_dir = '../transition_states'
    os.chdir(ts_dir)

    log_files = glob('*.log')

    ts_files = []

    for file in log_files:
        if (ts_name[:-4] in file) and ('optts' in file):
            ts_files.append(file)

    if len(ts_files) == 1:
        shutil.copy(f"{ts_files[0]}", f"{dst_dir}/frequency_logs/{file.split('_')[0]}_{idx}_optts_g16.log")
    else:
        for ts in ts_files:
            freq_log = freq_from_gaussian(ts)
            if freq_log is not None:
                if abs(freq_log - freq_xyz) < 0.5:
                    shutil.copy(f"{ts}", f"{dst_dir}/frequency_logs/{file.split('_')[0]}_{idx}_optts_g16.log")
    return None
    
def freq_from_gaussian(ts_file : str) -> float:
    """ Get the imaginary freq of the gaussian output """

    with open(ts_file, 'r') as out_gaussian:
        out_lines = out_gaussian.readlines()

    low_freqs = []

    for line in out_lines:

        if "Low frequencies ---" in line:
            low_freqs.append(line)
            break

    if low_freqs:
        frequency = float(low_freqs[0][20:].split()[0])
        return frequency
    
    else:
        return None
        
def copy_sp_files(idx : str, dst_dir : str) -> None:
    """ Copy the sp outputs of autode to a new directory"""

    sp_dir = '../single_points'
    os.chdir(sp_dir)

    log_files = glob('*.log')

    [shutil.copy(f"{file}", f"{dst_dir}/single_points_logs/{file.split('_')[0]}_{idx}_sp_g16.log") for file in log_files]

    return None

def copy_freq_files(idx : str, dst_dir : str) -> None:
    """ Copy the hessian outputs of autode to a new directory"""

    freq_dir = '../thermal'
    os.chdir(freq_dir)

    log_files = glob('*.log')

    [shutil.copy(f"{file}", f"{dst_dir}/frequency_logs/{file.split('_')[0]}_{idx}_hess_g16.log") for file in log_files]

    return None 

if __name__ == "__main__":
        copy_imp_files()
