import pandas as pd


def filter_data():

    df = pd.read_csv('original_data/drug_bdes_paton.csv', index_col=0)

    # models for Cpd I of cytochrome P450
    rads, mols = ['C[O]', '[O]c1ccccc1', '[O]c1ccc(N=O)cc1'], ['CO', 'Oc1ccccc1', 'O=Nc1ccc(O)cc1']

    data = []

    for row in df.itertuples():
        for rad, mol in zip(rads, mols):
            rxn = f"{row.molecule}.{rad}>>{mol}.{row.fragment2}"
            data.append((row.mol_id, rxn, row.is_metab_site))

    reactivity_dataset = pd.DataFrame(data, columns=['rxn_id', 'rxn_smiles', 'is_metab_site'])
    reactivity_dataset.to_csv('paton_test.csv')

    smiles = []

    for row in reactivity_dataset.itertuples():
        r1, r2, p1, p2 = reacs_prods(row.rxn_smiles)
        smiles.append(r1)
        smiles.append(r2)
        smiles.append(p1)
        smiles.append(p2)

    smiles = list(set(smiles))
    df_smiles = pd.DataFrame(smiles, columns=['smiles'])
    df_smiles.to_csv('species_reactivity_dataset_paton_drugs.csv')


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
