import os
from datetime import datetime

import pandas as pd
from pymer4.models import Lmer

from coco_hothand.config import data_root, lme4_optimizers
from coco_hothand.figures.util.regression import export_regression_table, prepare_result
from coco_hothand.util.stats_util.correct_dv import correct_dv

condition = 'solist_solist'
var_valid = 'valid_new_max2'
dv = 'regquality'
var_ranS = 'prev_won'
cols = [
    'subject',  dv, var_valid, 'tdiff_new_max2', 'prev_won',
    'rule_kurze', 'game_type', 'cur_pos'
]
df = pd.read_parquet(os.path.join(data_root, f'{condition}.parquet'), columns=cols, engine='fastparquet')
df = df[df[var_valid] & (df.tdiff_new_max2 < 900)]

regs = [
    'prev_won',
    'C(rule_kurze)',
    'C(game_type)',
    'C(cur_pos)'
]

print(f'N={len(df)}')

reload = False
filename = __file__.split('/')[-1].replace('.py', '.parquet')
if reload:
    patsy = f"{dv} ~ {' + '.join(regs)} + (1+{var_ranS}|subject)"
    # 'nlopt_bobyqa', 'nlopt_neldermead', 'nlminb', 'nmkbw', 'bobyqa', 'neldermead', 'lbfgsb'
    optimizer = 'nlopt_bobyqa'  # nlopt_bobyqa is the default optimizer!
    model = Lmer(patsy, data=df[[dv, 'subject'] + [reg.replace('C(', '').replace(')', '') for reg in regs]])

    print(f'[{datetime.now().strftime('%H:%M:%S')}] Performing Lmer analysis [{optimizer}]: {patsy}')
    model.fit(summary=False, control=lme4_optimizers[optimizer])
    print(f'\t[{datetime.now().strftime('%H:%M:%S')}] ... Lmer finished')
    result = prepare_result(model)
    result.attrs['optimizer'] = optimizer
    result.to_parquet(os.path.join('../data', filename))
else:
    result = pd.read_parquet(os.path.join('../data', filename))

print(f'\nOptimizer: {result.attrs['optimizer']}')
print(f"\nCorrelation matrix:\n{df[[dv] + [reg.replace('C(', '').replace(')', '') for reg in regs if reg != var_ranS]].corr()}")

ff_cor = correct_dv(df, dv, result)
ff_cor.to_parquet(f"../data/figures/{__file__.split('/')[-1].replace('stats', 'prepare_figure').replace('.py', '_corrected.parquet')}")
print('\nDependent variable:\n', ff_cor)

export_regression_table(result, __file__)

print(f'Effect lme4: {result.iloc[1]['Estimate']:.5f}  ||  Effect correction: {ff_cor.loc['prev_won_cor', 'av'] - ff_cor.loc['prev_lost_cor', 'av'] :.5f}')