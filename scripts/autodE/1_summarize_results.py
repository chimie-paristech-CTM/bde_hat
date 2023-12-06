#!/usr/bin/python

import os
from glob import glob
import pandas as pd
import shutil


def summarize_autode_output():
    """ From the autodE folder, summarize all the results """

    os.chdir('../../autodE_input/0.autode_resume_dir')
    folders = glob('rxn*')
    
    autode_results = pd.DataFrame(columns=['ID', 'RXN_SMILES', 'dG_act', 'dG_rxn', 'time'])

    current_dir = os.getcwd()

    for folder in folders:
        
        folder_dir = os.path.join(current_dir, folder)
        os.chdir(folder_dir)
        energies_file = glob('*.csv')

        if not energies_file:
            continue
            
        if check_copies():
            with open('../errors_copy.txt', 'a') as file:
                file.write(f"{folder}\n")
            continue

        idx = folder[4:]
        change_name(idx)
        time = extract_time()
        rxn_smile = extract_rxn_smile()
        dG_act, dG_rxn = extract_energy(energies_file[0])
        rxn_info = [idx, rxn_smile, dG_act, dG_rxn, time]
        autode_results = pd.concat([autode_results, pd.DataFrame([rxn_info], columns=['ID', 'RXN_SMILES', 'dG_act', 'dG_rxn', 'time'])], ignore_index=True)

    os.chdir(current_dir)
    autode_results['frequency'] = autode_results['ID'].apply(
        lambda x: frequency(f"rxn_{x:07}/frequency_logs/TS_{x:07}_optts_g16.log"))
    autode_results.to_csv('autode_results.csv')

    return None

def check_copies() -> bool:

    xyz_files = glob('*.xyz')
    energy_file = glob('*.csv')

    os.chdir('frequency_logs')
    freq_files = glob('*.log')

    os.chdir('../single_points_logs')
    sp_files = glob('*.log')

    files = len(sp_files) + len(freq_files) + len(energy_file) + len(xyz_files)

    os.chdir('../')

    return True if files != 17 else False   # 17 because 6 .xyz + 1 .csv + 5 freq .log + 5 sp .log


def extract_time():
    """ Extract time calculation """
    
    out_time_file = glob('*.out')

    with open(out_time_file[0], 'r') as time_file:
        time_line = time_file.readlines()[0]
    
    time = float(time_line.split()[1])/3600

    return time


def extract_rxn_smile():
    """ Extract rxn smile """
    
    with open('rxn_smile.txt', 'r') as rxn_file:
        rxn_smile = rxn_file.readlines()[0] 

    return rxn_smile


def extract_energy(energies_file):
    """ From energy.csv return Free Gibbs Energy of Activation and of Reaction"""
    
    prods = []
    reacs = []

    energies = pd.read_csv(energies_file)

    for specie in energies.Species:

        if specie.startswith('r'):
            reacs.append(energies.loc[energies.Species == specie].index[0])
        elif specie.startswith('p'):
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
    
    dG_rxn = (G_prods - G_reacs)*627.509
    dG_act = (G_ts - G_reacs)*627.509

    return dG_act, dG_rxn


def change_name(idx):
    """ some TS has different name because the autodE generated in that way, so we have to fix it."""
    
    pwd = os.getcwd()
    os.chdir(f"frequency_logs")

    file = glob('*optts_g16.log')

    if not file[0].startswith('TS_'):
        shutil.copy(file[0], f"TS_{idx}_optts_g16.log")
        os.remove(file[0])
    
    os.chdir(pwd)
    
    return None


def frequency(file):
    """ Read gaussian output for frequency """

    with open(file, 'r') as g16_output:
        lines = g16_output.readlines()

    for line in lines:
        if "Low frequencies ---" in line:
            break

    freq = float(line[20:].split()[0])
    return freq


if __name__ == "__main__":
    summarize_autode_output()
