import os
from datetime import datetime
import socket

import pandas as pd
from pymer4.models import Lmer

from coco_hothand.config import data_root, lme4_optimizers
from coco_hothand.figures.util.regression import export_regression_table, prepare_result
from coco_hothand.util.stats_util.correct_dv import correct_dv
from coco_hothand.util.stats_util.expand_predictors import expand_predictors

if socket.gethostname() == 'kolja':
    os.environ["R_LIBS"] = "/usr/local/lib/R/site-library"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

gametype_dict = {0: 'ramsch', 1: 'rufspiel', 2: 'farbwenz', 3: 'geier', 4: 'wenz', 5: 'farbsolo'}

condition = 'solist_all'
var_valid = 'valid_new_max2'
dv = 'opponent_optimality_ratio'
var_ranS = 'prev_won'
cols = ['id',
    'subject', dv, var_valid, 'tdiff_new_max2', 'prev_won',
    'rule_kurze', 'game_type', 'cur_pos',
    'role', 'role_id',
    'solo'
]
df = pd.read_parquet(os.path.join(data_root, f'{condition}_os.parquet'), columns=cols, engine='fastparquet')
df = df[df[var_valid] & (df.tdiff_new_max2 < 900) & ~df[dv].isna()]
# df.loc[~df.player_role.isna(), 'role'] = df[~df.player_role.isna()].apply(lambda x: f"{gametype_dict[x['game_type']]}_{x['player_role']}", axis=1)

df['prev_won'] = df.prev_won - 0.5
# df['solo'] = df.solo - 0.5

regs = [
    'prev_won',
    'C(rule_kurze)',
    'C(role_id)',
    'C(cur_pos)'
]

print(f'N={len(df)}')
print(f'av time between failure and successes = {df.tdiff_new_max2.mean():.1f} seconds')

filename = __file__.split('/')[-1].replace('.py', '.parquet')
recompute = True
if recompute:
    patsy = f"{dv} ~ {' + '.join(regs)} + (1+{var_ranS}|subject)"
    # 'nlopt_bobyqa', 'nlopt_neldermead', 'nlminb', 'nmkbw', 'bobyqa', 'neldermead', 'lbfgsb'
    optimizer = 'nlopt_bobyqa'  # nlopt_bobyqa is the default optimizer!
    cols = [dv, 'subject'] + expand_predictors(regs)
    model = Lmer(patsy, data=df[cols])

    print(f'[{datetime.now().strftime('%H:%M:%S')}] Performing Lmer analysis [{optimizer}]: {patsy}')
    model.fit(summary=False, control=lme4_optimizers[optimizer])
    print(f'\t[{datetime.now().strftime('%H:%M:%S')}] ... Lmer finished')
    result = prepare_result(model)
    result.attrs['optimizer'] = optimizer
    result.to_parquet(os.path.join('data', filename))
else:
    result = pd.read_parquet(os.path.join('data', filename))

print(f'\nOptimizer: {result.attrs['optimizer']}')
# print(f"\nCorrelation matrix:\n{df[[dv] + [reg.replace('C(', '').replace(')', '') for reg in regs if reg != var_ranS]].corr()}")
print(f"\nCorrelation matrix:\n{df[[dv] + [reg for reg in regs if (reg != var_ranS) and not reg.startswith('C(')]].corr()}")

ff_cor = correct_dv(df, dv, result)
ff_cor.to_parquet(f"../data/figures/{__file__.split('/')[-1].replace('stats', 'prepare_figure').replace('.py', '_corrected.parquet')}")
print('\nDependent variable:\n', ff_cor)

export_regression_table(result, __file__)

print(f'Effect lme4: {result.iloc[1]['Estimate']:.5f}  ||  Effect correction: {ff_cor.loc['prev_won_cor', 'av'] - ff_cor.loc['prev_lost_cor', 'av'] :.5f}')