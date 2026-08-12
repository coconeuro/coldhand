import os
import socket
from datetime import datetime

import pandas as pd
from pymer4.models import Lmer

from coco_hothand.config import data_root, lme4_optimizers
from coco_hothand.figures.util.regression import export_regression_table, prepare_result
from coco_hothand.util.stats_util.correct_dv import correct_dv

if socket.gethostname() == 'kolja':
    os.environ["R_LIBS"] = "/usr/local/lib/R/site-library"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import rpy2.robjects as robjects
r_version = robjects.r('R.version.string')
print(r_version[0])

condition = 'solist_solist'
var_valid = 'valid_new_max2'
dv = 'opponent_optimality_ratio'
var_ranS = 'prev_won'
tmax = 900
cols = [
    'subject', var_valid,
    dv.replace('_z', '') if dv.endswith('_z') else dv,
    'tdiff_new_max2', 'prev_won',
    'rule_kurze', 'game_type', 'cur_pos',
    'position_solist',
    'player_augen', 'regquality_sologegner_cards_best',
    'regquality'
]
df = pd.read_parquet(os.path.join(data_root, f'{condition}.parquet'), columns=cols, engine='fastparquet')

# for col in cols:
#     df[col.replace('solist', 'player')] = df[col]
# df = df[[c for c in df.columns if c not in cols]]
df = df[df[var_valid] & (df.tdiff_new_max2 < tmax)]
df[dv] *= 100

# for i in range(4):
#     df.loc[df.position_solist == i, 'optimality'] = df[[f'player{j}_optimality_percentile' for j in range(4) if j != i]].mean(axis=1)

# for i in range(4):
#     df.loc[df.position_solist == i, 'optimality'] = df[f'player{i}_optimality_percentile']

df.loc[df[dv].isna(), dv] = 1
#
regs = [
    'prev_won',
    # 'regquality_sologegner_cards_best',
    'C(rule_kurze)',
    'C(game_type)',
    'C(cur_pos)'
]
print(f'N={len(df)}')
patsy = f"{dv} ~ {' + '.join(regs)} + (1+{var_ranS}|subject)"

recompute = True
filename = __file__.split('/')[-1].replace('.py', '.parquet')
if recompute:
    # 'nlopt_bobyqa', 'nlopt_neldermead', 'nlminb', 'nmkbw', 'bobyqa', 'neldermead', 'lbfgsb'
    optimizer = 'nlopt_bobyqa'  # nlopt_bobyqa is the default optimizer!
    model = Lmer(patsy, data=df[[dv, 'subject'] + [reg.replace('C(', '').replace(')', '') for reg in regs]])

    print(f'[{datetime.now().strftime('%H:%M:%S')}] Performing Lmer analysis [{optimizer}]: {patsy}')
    model.fit(summary=False, control=lme4_optimizers[optimizer])
    print(f'\t[{datetime.now().strftime('%H:%M:%S')}] ... Lmer finished')
    result = prepare_result(model)
    result.attrs['optimizer'] = optimizer
    result.attrs['tmax'] = tmax
    result.attrs['patsy'] = patsy
    result.attrs['nsamples'] = len(df)
    result.to_parquet(os.path.join('../data', filename))
else:
    result = pd.read_parquet(os.path.join('../data', filename))
    print(f"N={result.attrs['nsamples']}")

print(patsy)
print(f'\nOptimizer: {result.attrs['optimizer']}')
print(f"\nCorrelation matrix:\n{df[[dv] + [reg.replace('C(', '').replace(')', '') for reg in regs if reg != var_ranS]].corr()}")

ff_cor = correct_dv(df, dv, result)
ff_cor.to_parquet(f"../data/figures/{__file__.split('/')[-1].replace('stats', 'prepare_figure').replace('.py', '_corrected.parquet')}")
print('\nDependent variable:\n', ff_cor)

export_regression_table(result, __file__)

print(f'Effect lme4: {result.iloc[1]['Estimate']:.5f}  ||  Effect correction: {ff_cor.loc['prev_won_cor', 'av'] - ff_cor.loc['prev_lost_cor', 'av'] :.5f}')
