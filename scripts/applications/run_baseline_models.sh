#!/bin/bash

python3 baseline_models_omega.py
python3 baseline_models_omega.py --features 'dG_forward' 'dG_reverse'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1'
python3 baseline_models_omega.py --features 's_reac1' 's_prod0'
python3 baseline_models_omega.py --features 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'fr_dG_forward' 'fr_dG_reverse'
python3 baseline_models_omega.py --features 's_reac1' 's_prod0' 'dG_forward' 'dG_reverse'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'dG_forward' 'dG_reverse'
python3 baseline_models_omega.py --features 'dG_forward' 'dG_reverse' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'dG_forward' 'dG_reverse' 'fr_dG_forward' 'fr_dG_reverse'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 's_reac1' 'q_prod0' 's_prod0' 'q_prod1' 'qH_prod1'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'fr_dG_forward' 'fr_dG_reverse'
python3 baseline_models_omega.py --features 's_reac1' 's_prod0' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 's_reac1' 's_prod0' 'fr_dG_forward' 'fr_dG_reverse'
python3 baseline_models_omega.py --features 'fr_dG_forward' 'fr_dG_reverse'  'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'dG_forward' 'dG_reverse' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'dG_forward' 'dG_reverse' 'fr_dG_forward' 'fr_dG_reverse'
python3 baseline_models_omega.py --features 's_reac1' 's_prod0' 'dG_forward' 'dG_reverse' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 's_reac1' 's_prod0' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 's_reac1' 's_prod0' 'dG_forward' 'dG_reverse'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 's_reac1' 's_prod0' 'fr_dG_forward' 'fr_dG_reverse'
python3 baseline_models_omega.py --features 's_reac1' 's_prod0' 'fr_dG_forward' 'fr_dG_reverse' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'fr_dG_forward' 'fr_dG_reverse' 'BV_reac1' 'BV_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'dG_forward' 'dG_reverse' 'BV_reac1' 'BV_prod0' 's_reac1' 's_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'fr_dG_forward' 'fr_dG_reverse' 'BV_reac1' 'BV_prod0' 's_reac1' 's_prod0'
python3 baseline_models_omega.py --features 'q_reac0' 'qH_reac0' 'q_reac1' 'q_prod0' 'q_prod1' 'qH_prod1' 'fr_dG_forward' 'fr_dG_reverse' 'dG_forward' 'dG_reverse' 's_reac1' 's_prod0'


#python3 baseline_models_tantillo.py --features 's_rad' 'q_rad' 'Buried_Vol' 'BDFE' 'fr_BDE'

#python3 baseline_models_tantillo.py --features 's_rad'

#python3 baseline_models_tantillo.py --features 'q_rad'

#python3 baseline_models_tantillo.py --features 'Buried_Vol'

#python3 baseline_models_tantillo.py --features 'BDFE'

#python3 baseline_models_tantillo.py --features 'fr_BDE'

#python3 baseline_models_tantillo.py --features 's_rad' 'BDFE'

#python3 baseline_models_tantillo.py --features 's_rad' 'BDFE'

#python3 baseline_models_tantillo.py --features 's_rad' 'q_rad'

#python3 baseline_models_tantillo.py --features 's_rad' 'fr_BDE'

#python3 baseline_models_tantillo.py --features 's_rad' 'Buried_Vol'

#python3 baseline_models_tantillo.py --features 'BDFE' 'q_rad'

#python3 baseline_models_tantillo.py --features 'BDFE' 'fr_BDE'

#python3 baseline_models_tantillo.py --features 'BDFE' 'Buried_Vol'

#python3 baseline_models_tantillo.py --features 'q_rad' 'fr_BDE'

#python3 baseline_models_tantillo.py --features 'q_rad' 'Buried_Vol'

#python3 baseline_models_tantillo.py --features 'Buried_Vol' 'fr_BDE'

#python3 baseline_models_tantillo.py --features 's_rad' 'Buried_Vol' 'q_rad'

#python3 baseline_models_tantillo.py --features 's_rad' 'Buried_Vol' 'fr_BDE'

#python3 baseline_models_tantillo.py --features 's_rad' 'Buried_Vol' 'BDFE'