import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def histogram_act_rxn_autodE():
    """ histogram for reactivity database"""

    df = pd.read_csv('../../data/reactivity_database_mapped.csv', index_col=0)

    mean_act = df.DG_TS.mean()
    std_act = df.DG_TS.std()
    mean_act_tunn = df.DG_TS_tunn.mean()
    std_act_tunn = df.DG_TS_tunn.std()
    mean_rxn = df.G_r.mean()
    std_rxn = df.G_r.std()

    r_act = df.corr()['G_r'].loc['DG_TS']
    r_tunn = df.corr()['G_r'].loc['DG_TS_tunn']

    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    sns.histplot(df.DG_TS, ax=axs[0, 0], color='darkred')
    sns.histplot(df.DG_TS_tunn, ax=axs[0, 1], color='darkred')
    sns.histplot(df.G_r, ax=axs[0, 2], color='darkred')
    sns.kdeplot(data=df, x='G_r', y='DG_TS', ax=axs[1, 0], cmap='rocket_r', fill=True, thresh=0)
    sns.kdeplot(data=df, x='G_r', y='DG_TS_tunn', ax=axs[1, 1], cmap='rocket_r', fill=True, thresh=0)

    axs[0, 0].text(0.02, 0.9, f"mean: {mean_act:.3f}\nstd: {std_act:.3f}", transform=axs[0, 0].transAxes, fontsize=12)
    axs[0, 1].text(0.02, 0.9, f"mean: {mean_act_tunn:.3f}\nstd: {std_act_tunn:.3f}", transform=axs[0, 1].transAxes, fontsize=12)
    axs[0, 2].text(0.02, 0.9, f"mean:  {mean_rxn:.3f}\nstd: {std_rxn:.3f}", transform=axs[0, 2].transAxes, fontsize=12)
    axs[1, 0].text(0.02, 0.95, f"R$^2$ = {r_act ** 2:.2f}", transform=axs[1, 0].transAxes, fontsize=12)
    axs[1, 1].text(0.02, 0.95, f"R$^2$ = {r_tunn ** 2:.2f}", transform=axs[1, 1].transAxes, fontsize=12)

    axs[0, 0].set_xlabel('$\Delta$G$^{\ddag}$ (kcal/mol)', fontsize=12)
    axs[0, 1].set_xlabel('$\Delta$G$^{\ddag}$ tunneling (kcal/mol)', fontsize=12)
    axs[0, 2].set_xlabel('$\Delta$G$_{rxn}$ (kcal/mol)', fontsize=12)
    axs[1, 0].set_xlabel('$\Delta$G$_{rxn}$ (kcal/mol)', fontsize=12)
    axs[1, 0].set_ylabel('$\Delta$G$^{\ddag}$  (kcal/mol)', fontsize=12)
    axs[1, 1].set_xlabel('$\Delta$G$_{rxn}$ (kcal/mol)', fontsize=12)
    axs[1, 1].set_ylabel('$\Delta$G$^{\ddag}$ tunneling (kcal/mol)', fontsize=12)

    plt.delaxes(axs[1, 2])
    plt.tight_layout()
    plt.savefig('histogram_act_rxn_autodE')

    return None


def correlation_plot_ffnn():

    k_fold = 9
    df = pd.DataFrame()
    df_results = pd.read_csv('../../data/reactivity_database_mapped.csv', index_col=0)

    for i in range(k_fold):
        df_k_fold = pd.read_csv(f'results_ffnn/test_predicted_{i}.csv', index_col=0)
        df = pd.concat([df, df_k_fold])

    df_result_test = df_results.loc[df_results['rxn_id'].isin(df['rxn_id'])]

    df = df.merge(df_result_test[['rxn_id', 'DG_TS']], on='rxn_id')

    r2 = r2_score(df.DG_TS, df.predicted_activation_energy)

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    sns.kdeplot(data=df, x='DG_TS', y='predicted_activation_energy', cmap='rocket_r', fill=True, thresh=0)

    axs.text(0.02, 0.95, f"R$^2$ = {r2:.2f}", transform=axs.transAxes, fontsize=12)

    axs.set_title('$\Delta$G$^{\ddag}$ (kcal/mol)')
    axs.set_xlabel('computed $\Delta$G$^{\ddag}$ (kcal/mol)')
    axs.set_ylabel('predicted $\Delta$G$^{\ddag}$ (kcal/mol)')

    plt.tight_layout()
    plt.savefig('correlation_between_predicted_computed')


def checking_frequencies():

    ffnn_values = [(0, 2.445, 3.760, 1241),
                   (500, 2.370, 3.548, 1193),
                   (750, 2.352, 3.520, 1173),
                   (1000, 2.326, 3.474, 1141),
                   (1150, 2.259, 3.315, 1114),
                   (1300, 2.255, 3.346, 1114),
                   (1400, 2.193, 3.145, 993),
                   (1500, 2.243, 3.170, 808),
                   (1600, 2.457, 3.382, 485)]

    data = pd.DataFrame()
    data[['freq', 'mae_act', 'rmse_act', 'samples']] = ffnn_values

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    sns.lineplot(x='freq', y='mae_act', data=data, marker='o', ax=axes[0], color='darkred')
    sns.lineplot(x='freq', y='rmse_act', data=data, marker='o', ax=axes[1], color='darkred')

    axes[0].set_xlabel('frequency (cm$^{-}$)')
    axes[0].set_ylabel('$\Delta$G$^{\ddag}$ MAE (kcal/mol)')
    axes[1].set_xlabel('frequency (cm$^{-}$)')
    axes[1].set_ylabel('$\Delta$G$^{\ddag}$ RMSE (kcal/mol)')

    # remove grid lines
    axes[0].grid(False)
    axes[1].grid(False)

    plt.tight_layout()
    plt.savefig('errors_frequencies')


def learning_curves_ffnn_and_ensembles():

    ffnn_values = [(200, 3.71, 5.18, 1, 2.46, 3.65),
                   (400, 2.79, 4.07, 3, 2.42, 3.62),
                   (800, 2.46, 3.68, 5, 2.40, 3.60),
                   (1000, 2.42, 3.63, 7, 2.41, 3.62),
                   (1228, 2.39, 3.59, 10, 2.39, 3.59)]

    data = pd.DataFrame()
    data[['samples', 'mae_act_s', 'rmse_act_s', 'ensembles', 'mae_act_e', 'rmse_act_e']] = ffnn_values

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    sns.lineplot(x='samples', y='mae_act_s' , data=data, marker='o', ax=axes[0, 0], color='darkred')
    sns.lineplot(x='samples', y='rmse_act_s', data=data, marker='o', ax=axes[0, 1], color='darkred')
    sns.lineplot(x='ensembles', y='mae_act_e', data=data, marker='o', ax=axes[1, 0], color='darkred')
    sns.lineplot(x='ensembles', y='rmse_act_e', data=data, marker='o', ax=axes[1, 1], color='darkred')

    axes[0, 0].set_xlabel('samples')
    axes[0, 0].set_ylabel('MAE $\Delta$G$^{\ddag}$ (kcal/mol)')
    axes[0, 1].set_xlabel('samples')
    axes[0, 1].set_ylabel('RMSE $\Delta$G$^{\ddag}$ (kcal/mol)')

    axes[1, 0].set_xlabel('No. ensembles')
    axes[1, 0].set_ylabel('MAE $\Delta$G$^{\ddag}$ (kcal/mol)')
    axes[1, 1].set_xlabel('No. ensembles')
    axes[1, 1].set_ylabel('RMSE $\Delta$G$^{\ddag}$ (kcal/mol)')

    # remove grid lines
    axes[0, 0].grid(False)
    axes[0, 1].grid(False)
    axes[1, 0].grid(False)
    axes[1, 1].grid(False)

    plt.tight_layout()
    plt.savefig('learning_curves_and_ensembles_ffnn')


def learning_curves_gnn():

    gnn_values = [(200, 3.09, 4.54, 2.96, 4.41, 3.66, 5.28, 3.51, 5.12),
                  (400, 2.76, 4.01, 2.59, 3.79, 3.40, 4.92, 3.25, 4.74),
                  (600, 2.65, 3.85, 2.47, 3.64, 3.28, 4.71, 3.13, 4.53),
                  (800, 2.56, 3.77, 2.37, 3.55, 3.20, 4.59, 3.05, 4.41),
                  (1000, 2.52, 3.79, 2.34, 3.56, 3.19, 4.58, 3.03, 4.39),
                  (1228, 2.52, 3.75, 2.29, 3.46, 3.06, 4.42, 2.93, 4.25)]

    data = pd.DataFrame()
    data[['samples', 'mae_act', 'rmse_act', 'mae_act_tunn', 'rmse_act_tunn',
          'mae_act_gnn', 'rmse_act_gnn', 'mae_act_tunn_gnn', 'rmse_act_tunn_gnn']] = gnn_values

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    sns.lineplot(x='samples', y='mae_act', data=data, marker='o', ax=axes[0, 0], label='qm-GNN', color='darkred')
    sns.lineplot(x='samples', y='rmse_act', data=data, marker='o', ax=axes[0, 1], label='qm-GNN', color='darkred')
    sns.lineplot(x='samples', y='mae_act_gnn', data=data, marker='o', ax=axes[0, 0], label='GNN', color='darkblue')
    sns.lineplot(x='samples', y='rmse_act_gnn', data=data, marker='o', ax=axes[0, 1], label='GNN', color='darkblue')
    sns.lineplot(x='samples', y='mae_act_tunn', data=data, marker='o', ax=axes[1, 0], label='qm-GNN', color='darkred')
    sns.lineplot(x='samples', y='rmse_act_tunn', data=data, marker='o', ax=axes[1, 1], label='qm-GNN', color='darkred')
    sns.lineplot(x='samples', y='mae_act_tunn_gnn', data=data, marker='o', ax=axes[1, 0], label='GNN', color='darkblue')
    sns.lineplot(x='samples', y='rmse_act_tunn_gnn', data=data, marker='o', ax=axes[1, 1], label='GNN', color='darkblue')

    axes[0, 0].set_xlabel('samples')
    axes[0, 0].set_ylabel('MAE $\Delta$G$^{\ddag}$ (kcal/mol)')
    axes[0, 1].set_xlabel('samples')
    axes[0, 1].set_ylabel('RMSE $\Delta$G$^{\ddag}$ (kcal/mol)')

    axes[1, 0].set_xlabel('samples')
    axes[1, 0].set_ylabel('MAE $\Delta$G$^{\ddag}$ tunneling (kcal/mol)')
    axes[1, 1].set_xlabel('samples')
    axes[1, 1].set_ylabel('RMSE $\Delta$G$^{\ddag}$ tunneling (kcal/mol)')

    # remove grid lines
    axes[0, 0].grid(False)
    axes[0, 1].grid(False)
    axes[1, 0].grid(False)
    axes[1, 1].grid(False)

    plt.tight_layout()
    plt.savefig('learning_curves_gnn')


def ensembles_gnn():

    values = [(1, 3.63, 5.10, 3.51, 4.94, 2.95, 4.21, 2.73, 3.96),
              (3, 3.27, 4.64, 3.09, 4.42, 2.66, 3.92, 2.45, 3.67),
              (5, 3.20, 4.63, 3.04, 4.43, 2.57, 3.82, 2.37, 3.56),
              (7, 3.16, 4.55, 3.02, 4.36, 2.59,  3.87, 2.41, 3.65),
              (10, 3.07, 4.41, 2.93, 4.25, 2.52, 3.75, 2.29, 3.46)]

    data = pd.DataFrame()
    data[['ensembles', 'mae_act_g', 'rmse_act_g', 'mae_act_gt', 'rmse_act_gt', 'mae_act_d', 'rmse_act_d', 'mae_act_dt', 'rmse_act_dt']] = values

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

    sns.lineplot(x='ensembles', y='mae_act_d', data=data, marker='o', ax=axes[0, 0], label='qm-GNN', color='darkred')
    sns.lineplot(x='ensembles', y='rmse_act_d', data=data, marker='o', ax=axes[0, 1], label='qm-GNN', color='darkred')
    sns.lineplot(x='ensembles', y='mae_act_g', data=data, marker='o', ax=axes[0, 0], label='GNN', color='darkblue')
    sns.lineplot(x='ensembles', y='rmse_act_g', data=data, marker='o', ax=axes[0, 1], label='GNN', color='darkblue')
    sns.lineplot(x='ensembles', y='mae_act_gt', data=data, marker='o', ax=axes[1, 0], label='GNN', color='darkblue')
    sns.lineplot(x='ensembles', y='rmse_act_gt', data=data, marker='o', ax=axes[1, 1], label='GNN', color='darkblue')
    sns.lineplot(x='ensembles', y='mae_act_dt', data=data, marker='o', ax=axes[1, 0], label='qm-GNN', color='darkred')
    sns.lineplot(x='ensembles', y='rmse_act_dt', data=data, marker='o', ax=axes[1, 1], label='qm-GNN', color='darkred')

    axes[0, 0].set_xlabel('No. ensembles')
    axes[0, 0].set_ylabel('MAE $\Delta$G$^{\ddag}$ (kcal/mol)')
    axes[0, 1].set_xlabel('No. ensembles')
    axes[0, 1].set_ylabel('RMSE $\Delta$G$^{\ddag}$ (kcal/mol)')

    axes[1, 0].set_xlabel('No. ensembles')
    axes[1, 0].set_ylabel('MAE $\Delta$G$^{\ddag}$ (kcal/mol)')
    axes[1, 1].set_xlabel('No. ensembles')
    axes[1, 1].set_ylabel('RMSE $\Delta$G$^{\ddag}$ (kcal/mol)')

    # remove grid lines
    axes[0, 0].grid(False)
    axes[0, 1].grid(False)
    axes[1, 0].grid(False)
    axes[1, 1].grid(False)

    plt.tight_layout()
    plt.savefig('ensembles_gnn')


def correlation_between_frozen_BDFE():

    paton_rxns = pd.read_csv('../../data/paton_rxns_frozen_BDFE.csv')
    paton_rxns.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    old_bdfe = pd.read_csv('../../../scripts_morfeus/paton_rxns_fr.csv')
    old_bdfe.rename(columns={'Unnamed: 0': 'ID'}, inplace=True)
    old_bdfe = old_bdfe.loc[old_bdfe['BDFE_fr'] < 1000]
    df = old_bdfe.loc[old_bdfe['ID'].isin(paton_rxns['ID'])]
    df = pd.merge(df, paton_rxns, on='ID', suffixes=('_SP', '_opt'))
    df_corr = df.corr()
    r2 = (df_corr.loc['BDFE_fr_SP'].BDFE_fr_opt)**2

    fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(5, 5))
    sns.kdeplot(data=df, x='BDFE_fr_SP', y='BDFE_fr_opt', cmap='rocket_r', fill=True, thresh=0)

    axs.text(0.02, 0.95, f"R$^2$ = {r2:.2f}", transform=axs.transAxes, fontsize=12)

    axs.set_title('frozen BDE (kcal/mol)')
    axs.set_xlabel('single point frozen BDE (kcal/mol)')
    axs.set_ylabel('opt frozen BDE (kcal/mol)')

    plt.tight_layout()
    plt.savefig('correlation_between_frozen_sp_opt')



