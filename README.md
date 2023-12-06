# Hydrogen atom transfer reaction dataset
This repository contains the code and auxiliary data associated to the hydrogen atom transfer reaction dataset construction project. Code is provided 
"as-is". Minor edits may be required to tailor the scripts for different computational systems.

### Conda environment
To set up the conda environment:
```
conda env create --name <env-name> --file environment.yml
```

### Requirements
Comments about some libraries installed

1. [BDE](https://github.com/pstjohn/bde): Extracted and analyse radical dataset
2. [Pebble](https://github.com/noxdafox/pebble): Multiprocessing
3. [DRFP](https://github.com/reymond-group/drfp): Fingerprints generation
4. [MORFEUS](https://github.com/digital-chemistry-laboratory/morfeus): Calculation of buried volumes
5. [autodE](https://github.com/duartegroup/autodE): Calculation of reaction profiles.

Additionally, in order to execute the autodE high-throughput reaction profile computation workflow, Gaussian09/Gaussian16 and xTB needs to be accessible. 
More information about the autodE module can be found [here](https://github.com/duartegroup/autodE).

## Generating the search space and reaction SMILES

The generation of the four datasets can be found in the `scripts/gen_datasets` directory. For the generation of the HAT chemical space, is necessary the
[radical database](https://doi.org/10.6084/m9.figshare.c.4944855.v1) compiled by [St. John et al.](https://doi.org/10.1038/s41597-020-00588-x) in the 
`data` directory. Generating the chemical space and sampling a subset of representative reactions is done by:
```
python generation_HAT.py
```

By default, the script generates several `.csv` files in the `data` directory, the main ones are:

1. `reactions_2k_autodE.csv`, input for the reaction profile generation workflow.
2. `subset_30_g16_xtb_autodE.csv`, input for benchmarking.
3. `reactions_1M.csv`, the chemical space.

## High-throughput reaction profile computation

Input files for high-throughput reaction profile computations, based on the reaction SMILES outputted in the previous section, can be generated 
with the help of the `0.calculations.py` script in the `scripts/autodE` directory as follows:

```
python 0.calculations.py --final_dir <name of the autodE folder to be generated> --csv_file <path to the input .csv file> [--conda_env <conda environment of autodE package>]
```

The script batches up the reaction SMILES in individual Python files, each containing a linear sequence of autodE reaction profile computations, 
and places them in the `autodE_input`. Additionally, the script generates sub-directories for each individual reaction SMILES in the
`autodE_input` (the sub-directories are numbered based on the indices in the input file). By default, 24 cores are used per computation and the 
M06-2X/def2tzvp//M06-2X/def2svp level of theory is used. These default options can be adjusted modifying the script.

Under the condition that the scripts in the previous section have been run, the `0.calculations.py` script can for example be executed as follows:
```
python 0_calculations.py --csv_file ../../data/reactions_2k_autodE.csv --final_dir ../../autodE_input
```

The `autodE` directory also contains a script to extract the relevant output from the `autodE_folder`. This script 
can be executed as follows:
```
python aux_script.py
```

In our workflow, the `aux_script.py` is copied into each sub-directories and executed once the autodE finished. This script will copy all 
relevant files (.log, .xyz, .csv) to a new directory (`0.autode_resume_dir/<rxn_###>`). 

To summarize all the results in a final .csv file, the `1_summarize_results.py` script is provided. It creates a .csv file containing the successfully 
computed reaction SMILES together with the activition and reaction free energies(`autode_results.csv`). Execution as follow:
```
python 1_summarize_results.py
```

Examples of this can be found in the `autodE_input` directory.

Note that some reactions require postprocessing to make the geometry of the products compatible with the selected TS conformer or 
to avoid some wrong TS. The scripts below have also been included in this repository to facilitate this postprocessing.

## Post-processing reaction SMILES

## Tunneling correction

The script for calculation of tunneling correction can be found in the same `scripts/autodE` directory and is `3_eckart_potential`. This 
script is an adaptation of this [repo](https://github.com/SJ-Ang/PyTUN). Execution as follow:
```
python 3_eckart_potential.py
```

The script takes all the necessary information from the autodE resume directory(`autodE_input/0.autode_resume_dir`), the input is the 
final .csv file of the post-processing step(`reactivity_database.csv`) and the output is `reactivity_database_corrected.csv`, both files
can be found in the `data` directory.

## Baseline ML models

All the files related to the baseline models are included in the `script/baseline_models` directory. The baseline_model.py script, 
which runs each of the baseline models sequentially, can be executed as follows:
````
python baseline_models.py [--csv-file <path to the file containing the rxn-smiles>] [--input-file <path to the .pkl input file>] [--split-dir <path to the folder containing the requested splits for the cross validation>] [--n-fold <the number of folds to use during cross validation'>] [--features <features for models based in descriptors>]
````

For models based in fingerprints is only necessary the `csv-file` an for the models based in descriptors is only necessary 
the `input-file`. The fingerprints are generated during the run and the [DRFP](https://doi.org/10.1039/D1DD00006C) fingerprint is used.

A version of both input files are included in the ``data`` directory: `reactivity_database_mapped.csv` and `input_ffnn.pkl`. To generate an input file from scratch, 
you should use this [repo](https://github.com/chimie-paristech-CTM/energy_predictor_HAT).

## References

If (parts of) this workflow are used as part of a publication please cite the associated paper:
```
@article{hat_predictor,
         title={Repurposing QM Descriptor Datasets for on the Fly Generation of Informative Reaction Representations: 
         Application to Hydrogen Atom Transfer Reactions}, 
         author={Javier E. Alfonso-Ramos, Rebecca M. Neeser and Thijs Stuyver},
         journal="{ChemRxiv}",
         year="{2023}",
         doi="--",
         url="--"
}
```

Additionally, since the workflow makes heavy use of autodE, please also cite the paper in which this code was originally presented:
```
@article{autodE,
  doi = {10.1002/anie.202011941},
  url = {https://doi.org/10.1002/anie.202011941},
  year = {2021},
  publisher = {Wiley},
  volume = {60},
  number = {8},
  pages = {4266--4274},
  author = {Tom A. Young and Joseph J. Silcock and Alistair J. Sterling and Fernanda Duarte},
  title = {{autodE}: Automated Calculation of Reaction Energy Profiles -- Application to Organic and Organometallic Reactions},
  journal = {Angewandte Chemie International Edition}
}
```
