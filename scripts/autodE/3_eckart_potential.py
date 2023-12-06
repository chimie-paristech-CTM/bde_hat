import math
import os
from scipy.special import roots_legendre
import numpy as np
import pandas as pd

# https://pubs-acs-org.inc.bib.cnrs.fr/doi/epdf/10.1021/j100485a023
# Semiclassical tunneling calculations, Bruce C. Garrett and Donald G. Truhlar, J. Phys. Chem. 1979, 83, 22, 2921–2926

# https://github.com/SJ-Ang/PyTUN

# PHYSICAL CONSTANTS
GAS_CONSTANT = 8.3144621  # J/(K * mol)
PLANCK_CONSTANT = 6.62606957e-34  # J * s
BOLTZMANN_CONSTANT = 1.3806488e-23  # J/K
SPEED_OF_LIGHT = 2.99792458e10  # cm/s
AVOGADRO_CONSTANT = 6.0221415e23  # 1/mol
BOLTZMANN_CONSTANT_AU = 3.1668114E-6  # Boltzmann Constant in atomic units  Ha/K

# 10-point Gauss-Legendre Quadrature abscissa and weight
x, w = roots_legendre(10)


def reduced_mass(file):
    """ Read gaussian output for reduced mass"""
    with open(file, 'r') as g16_output:
        lines = g16_output.readlines()
    for line in lines:
        if line.startswith(' Red. masses --'):
            mu = line.split()[3]
            break
    return mu


def force_constant(file):
    """ Read gaussian output for force constant"""
    with open(file, 'r') as g16_output:
        lines = g16_output.readlines()
    for line in lines:
        if line.startswith(' Frc consts '):
            Fc = line.split()[3]
            break
    return Fc


def frequency(file):
    """ Read gaussian output for frequency """

    with open(file, 'r') as g16_output:
        lines = g16_output.readlines()

    for line in lines:
        if "Low frequencies ---" in line:
            break

    freq = float(line[20:].split()[0])
    return freq


def capital_b(V_max, V_r, V_p):  # eqn A3
    B = (np.sqrt(V_max) + np.sqrt(V_max - (V_p - V_r))) ** 2
    return B


def ALPHA(B, F_s, V_max, V_r, V_p):  # eqn A5
    dV = V_p - V_r
    alpha = (B * F_s / (2 * V_max * (V_max - dV))) ** 0.5
    return alpha


# Calculation of Transmission Probabilty of Eckart Potential
def transmission_probability(a, b, d):  # eqn A7
    trans_prob = (np.cosh(a + b) - np.cosh(a - b)) / (np.cosh(a + b) + np.cosh(d))
    return trans_prob


# Calculation of parameters a,b and d of Transmission Probabilty of Eckart Potential
def parameter_a(E, mu, alpha):  # eqn A8
    a = 2 * np.pi * np.sqrt(2 * mu * E) / alpha
    return a


def parameter_b(E, mu, V_p, V_r, alpha):  # eqn A9
    dV = V_p - V_r
    b = 2 * np.pi * np.sqrt(2 * mu * (E - dV)) / alpha
    return b


def parameter_d(B, mu, alpha):  # eqn A10
    d = 2 * np.pi * np.sqrt(abs((2 * mu * B - (alpha / 2) ** 2))) / alpha
    return d


# Calculation of SINH function of Kappa
def S(V_max, E, T):  # part eqn 12
    k = BOLTZMANN_CONSTANT_AU
    S = np.sinh(((V_max - E)) / (T * k))
    return S


def wigner_tunneling_correction(freq, temp=298.15): # Wigner tunneling approximation

    h = PLANCK_CONSTANT
    Kb = BOLTZMANN_CONSTANT
    c = SPEED_OF_LIGHT

    # convert freq to Hz because in gaussian is in cm^-1
    freq = freq * c

    kappa = 1 + (1/24)*((h * freq)/(Kb * temp))**2

    return kappa


def uncorrected_rate_constant(dG_act, temp=298.15): # rate constant using Eyring-Polany eqn and assuming transmission probability = 1

    h = PLANCK_CONSTANT
    Kb = BOLTZMANN_CONSTANT
    R = GAS_CONSTANT

    # Gaussian Hartree ... * 2625.5 [kJ/mol]  ... * 1000 [J/mol]
    dG_act = dG_act * 2625.5 * 1000

    kinetic_k = ((Kb * temp)/h)*math.exp(-dG_act/(R*temp))

    return kinetic_k


def corrected_dG(kappa, kinetic_k, temp=298.15):

    h = PLANCK_CONSTANT
    Kb = BOLTZMANN_CONSTANT
    R = GAS_CONSTANT

    dG_act_corr = R * temp * (math.log((Kb * temp)/(h * kappa * kinetic_k)))

    # convert to kcal/mol

    dG_act_corr = dG_act_corr/(1000 * 4.184)

    return dG_act_corr


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

    G_reacs = 0
    G_prods = 0
    E_reacs = 0
    E_prods = 0

    if len(prods) == 3:
        del prods[1]

    for prod in prods:
        G_prods += (energies[" E_sp"][prod] + energies[" G_cont"][prod])
        E_prods += energies[" E_sp"][prod]

    for reac in reacs:
        G_reacs += (energies[" E_sp"][reac] + energies[" G_cont"][reac])
        E_reacs += energies[" E_sp"][reac]

    G_ts = energies[" E_sp"][ts] + energies[" G_cont"][ts]
    E_ts = energies[" E_sp"][ts]

    dG_act = (G_ts - G_reacs)

    return dG_act, E_reacs, E_prods, E_ts


def zero_point_correction(file):
    with open(file, 'r') as g16_output:
        lines = g16_output.readlines()[::-1]
    for line in lines:
        if line.startswith(' Zero-point correction='):
            zpe = line.split()[2]
            break
    return float(zpe)


def eckart_tunneling_correction(V_max, E_o, V_r, V_p, frc_consts, mu, temp=298.15):

    k = BOLTZMANN_CONSTANT_AU

    # change of interval to [a, b] to [-1, 1]
    y = (V_max - E_o) / 2.0
    z = (V_max + E_o) / 2.0
    t = y * x + z
    B = capital_b(V_max, V_r, V_p)
    alpha = ALPHA(B, frc_consts, V_max, V_r, V_p)
    a = parameter_a(t, mu, alpha)
    b = parameter_b(t, mu, V_p, V_r, alpha)
    d = parameter_d(B, mu, alpha)
    s = S(V_max, t, temp)
    T = transmission_probability(a, b, d)
    sT = s * T
    integral = np.dot(w, sT) * y
    kappa = 1 + (2 * integral) / (k * temp)

    return kappa


def one_atom_rxn(idx):

    with open(f"rxn_{idx}/rxn_smile.txt", 'r') as file:
        rxn = file.readlines()[0]

    reacs = rxn.split('>>')
    reacs = [reac.split('.') for reac in reacs]
    reacs = [y for x in reacs for y in x]


    atoms = ['[Cl]', '[Br]', '[I]', '[F]']

    for reac in reacs:
        if reac in atoms:
            return reacs.index(reac)


def correcting_dG(idx):
    
    idx = f'{idx:07}'

    if idx[0] == '2':
        f_idx = f'0{idx[1:]}'
        ts_file = f"rxn_{f_idx}/frequency_logs/TS_{f_idx}_optts_g16.log"
        dG_act, E_reacs, E_prods, E_ts = extract_energy(f'corrected_rxns/rxn_{f_idx}/energies.csv')
        p_zpe = zero_point_correction(f"rxn_{f_idx}/frequency_logs/p0_{f_idx}_hess_g16.log") \
            + zero_point_correction(f"corrected_rxns/rxn_{f_idx}/frequency_logs/alt_p1_{f_idx}_hess_g16.log")
        r_zpe = zero_point_correction(f"rxn_{f_idx}/frequency_logs/r0_{f_idx}_hess_g16.log") \
                + zero_point_correction(f"rxn_{f_idx}/frequency_logs/r1_{f_idx}_hess_g16.log")
    else:
        ts_file = f"rxn_{idx}/frequency_logs/TS_{idx}_optts_g16.log"
        dG_act, E_reacs, E_prods, E_ts = extract_energy(f'rxn_{idx}/energies.csv')
        p_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/p0_{idx}_hess_g16.log") \
            + zero_point_correction(f"rxn_{idx}/frequency_logs/p1_{idx}_hess_g16.log")
        r_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/r0_{idx}_hess_g16.log") \
            + zero_point_correction(f"rxn_{idx}/frequency_logs/r1_{idx}_hess_g16.log")

    frc_consts = float(force_constant(ts_file)) / 15.569141  # mdyn/A to atomic units (https://david-hoffman.github.io/files/conversion_factors.pdf)
    mu = float(reduced_mass(ts_file)) * 1822.888486209  # mass electron
    freq = float(frequency(ts_file))

    # energy_sp + ZPE  (Hartree)

    E_r = E_reacs + r_zpe
    E_p = E_prods + p_zpe
    E_TS = E_ts + zero_point_correction(ts_file)


    E_o = max(E_r, E_p)
    V_max = E_TS
    V_r = E_r
    V_p = E_p

    # Scaling of Energies(define V_r == 0)
    V_max = V_max - V_r
    V_p = V_p - V_r
    E_o = E_o - V_r
    V_r = V_r - V_r

    if (V_max < 0) or (V_max - (V_p - V_r)) < 0:
        kappa = wigner_tunneling_correction(freq)
    else:
        kappa = eckart_tunneling_correction(V_max, E_o, V_r, V_p, frc_consts, mu)

    kinetic_k = uncorrected_rate_constant(dG_act, 298.15)

    dG_tunneling = corrected_dG(kappa, kinetic_k, 298.15)

    return dG_tunneling


def correcting_dG_rmechdb(idx):
    idx = f'{idx:07}'

    one_atom = one_atom_rxn(idx)

    ts_file = f"rxn_{idx}/frequency_logs/TS_{idx}_optts_g16.log"
    dG_act, E_reacs, E_prods, E_ts = extract_energy(f'rxn_{idx}/energies.csv')

    if one_atom == 2:
        p_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/p1_{idx}_hess_g16.log")
    elif one_atom == 3:
        p_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/p0_{idx}_hess_g16.log")
    else:
        p_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/p0_{idx}_hess_g16.log") \
            + zero_point_correction(f"rxn_{idx}/frequency_logs/p1_{idx}_hess_g16.log")

    if one_atom == 1:
        r_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/r0_{idx}_hess_g16.log")
    elif one_atom == 0:
        r_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/r1_{idx}_hess_g16.log")
    else:
        r_zpe = zero_point_correction(f"rxn_{idx}/frequency_logs/r0_{idx}_hess_g16.log") \
                + zero_point_correction(f"rxn_{idx}/frequency_logs/r1_{idx}_hess_g16.log")

    frc_consts = float(force_constant(
        ts_file)) / 15.569141  # mdyn/A to atomic units (https://david-hoffman.github.io/files/conversion_factors.pdf)
    mu = float(reduced_mass(ts_file)) * 1822.888486209  # mass electron
    freq = float(frequency(ts_file))

    # energy_sp + ZPE  (Hartree)

    E_r = E_reacs + r_zpe
    E_p = E_prods + p_zpe
    E_TS = E_ts + zero_point_correction(ts_file)

    E_o = max(E_r, E_p)
    V_max = E_TS
    V_r = E_r
    V_p = E_p

    # Scaling of Energies(define V_r == 0)
    V_max = V_max - V_r
    V_p = V_p - V_r
    E_o = E_o - V_r
    V_r = V_r - V_r

    if (V_max < 0) or (V_max - (V_p - V_r)) < 0:
        kappa = wigner_tunneling_correction(freq)
    else:
        kappa = eckart_tunneling_correction(V_max, E_o, V_r, V_p, frc_consts, mu)

    kinetic_k = uncorrected_rate_constant(dG_act, 298.15)

    dG_tunneling = corrected_dG(kappa, kinetic_k, 298.15)

    return dG_tunneling


if __name__ == "__main__":

    df = pd.read_csv('../../data/reactivity_database.csv')
    pwd = os.getcwd()
    os.chdir('../../autodE_input/0.autode_resume_dir')
    df['dG_act_corrected'] = df['rxn_id'].apply(lambda x: correcting_dG(x))
    os.chdir(pwd)
    df.to_csv('../../data/reactivity_database_corrected.csv')









