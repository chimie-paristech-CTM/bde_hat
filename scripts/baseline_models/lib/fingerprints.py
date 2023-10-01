#!/usr/bin/python
from drfp import DrfpEncoder
import pandas as pd
import numpy as np

def encode(smile_reaction : str, mapping_bool=False):
    """ Encode a smile reaction """

    return DrfpEncoder.encode(smile_reaction, mapping=mapping_bool)

def get_fingerprints_all_rxn_1(df):

    rxns = df.smiles.tolist()
    all_rxn_fps = []

    for rxn in rxns:
        fps = encode(rxn)
        all_rxn_fps.append(fps)
    
    dr = pd.DataFrame(np.array([item for s in all_rxn_fps for item in s]))
    dr[['rxn_id', 'G_r', 'DG_TS', 'DG_TS_tunn']] = df[['rxn_id', 'G_r', 'DG_TS', 'DG_TS_tunn']]

    return dr

def get_fingerprints_all_rxn(df):

    rxns = df.rxn_smiles.tolist()
    all_rxn_fps = []

    for rxn in rxns:
        fps = encode(rxn)
        all_rxn_fps.append(fps)
    
    dr = pd.DataFrame([s for s in all_rxn_fps], columns=['Fingerprints'])
    dr[['rxn_id', 'G_r', 'DG_TS', 'DG_TS_tunn']] = df[['rxn_id', 'G_r', 'DG_TS', 'DG_TS_tunn']]

    return dr
