import os
import pandas as pd
from datetime import datetime
from pymer4.models import Lmer

from coldhand.config import data_root, lme4_optimizers
from coldhand.figures.util.regression import export_regression_table, prepare_result
from coldhand.util.stats_util.correct_dv import correct_dv
from coldhand.util.stats_util.expand_predictors import expand_predictors

condition = 'solist_solist'
var_valid = 'new_max2'
dv = 'player_optimality_ratio'
var_ranS = 'prev_won'
tmax = 900
cols = [
    'id',
    'subject', dv, f'valid_{var_valid}', f'tdiff_{var_valid}', 'prev_won',
    'rule_kurze', 'game_type', 'cur_pos'
]
df = pd.read_parquet(os.path.join(data_root, f'{condition}_os.parquet'), columns=cols, engine='fastparquet')
df[dv] *= 100
df = df[df[f'valid_{var_valid}'] & (df[f'tdiff_{var_valid}'] < tmax)]

regs = [
    'prev_won',
    'C(rule_kurze)',
    'C(game_type)',
    'C(cur_pos)'
]
patsy = f"{dv} ~ {' + '.join(regs)} + (1+{var_ranS}|subject)"

print(f'N={len(df)}')
print(f'av time between failure and successes = {df[f'tdiff_{var_valid}'].mean():.1f} seconds')

filename = __file__.split('/')[-1].replace('.py', '.parquet')
recompute = False
if recompute:
    # 'nlopt_bobyqa', 'nlopt_neldermead', 'nlminb', 'nmkbw', 'bobyqa', 'neldermead', 'lbfgsb'
    optimizer = 'nlopt_bobyqa'  # nlopt_bobyqa is the default optimizer!
    cols = [dv, 'subject'] + expand_predictors(regs)
    model = Lmer(patsy, data=df[cols])

    print(f"[{datetime.now().strftime('%H:%M:%S')}] Performing Lmer analysis [{optimizer}]: {patsy}")
    model.fit(summary=False, control=lme4_optimizers[optimizer])
    print(f'\t[{datetime.now().strftime('%H:%M:%S')}] ... Lmer finished')
    result = prepare_result(model)
    result.attrs['optimizer'] = optimizer
    result.attrs['tmax'] = tmax
    result.attrs['patsy'] = patsy
    result.attrs['nsamples'] = len(df)
    result.to_parquet(os.path.join('data', filename))
else:
    result = pd.read_parquet(os.path.join('data', filename))

print(f'\nOptimizer: {result.attrs['optimizer']}')
print(f"\nCorrelation matrix:\n{df[[dv] + [reg.replace('C(', '').replace(')', '') for reg in regs if reg != var_ranS]].corr()}")

ff_cor = correct_dv(df, dv, result)
ff_cor.to_parquet(f"../data/figures/{__file__.split('/')[-1].replace('stats', 'prepare_figure').replace('.py', '_corrected.parquet')}")
print('\nDependent variable:\n', ff_cor)

export_regression_table(result, __file__, print_google_doc_size=True, exact_p=True)
print(patsy)

print(f'Effect lme4: {result.iloc[1]['Estimate']:.5f}  ||  Effect correction: {ff_cor.loc['prev_won_cor', 'av'] - ff_cor.loc['prev_lost_cor', 'av'] :.5f}')