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

condition = 'solist_allplayed'
var_valid = 'valid_new_max2'
dv = 'performance'
var_ranS = 'prev_won'
cols = [
    'id',
    'subject', var_valid, 'tdiff_new_max2', 'prev_won',
    'rule_kurze', 'game_type', 'cur_pos',
    'role', 'role_id',
    'player_augen', 'regquality', 'regquality_sologegner_cards_best'
]

# df = pd.read_parquet(os.path.join(data_root, f'{condition}.parquet'), engine='fastparquet')
# df.loc[~df.player_role.isna(), 'role'] = df[~df.player_role.isna()].apply(lambda x: f"{gametype_dict[x['game_type']]}_{x['player_role']}", axis=1)
# roles = ('rufspiel_sauspieler', 'rufspiel_mitspieler', 'rufspiel_sauspielgegner', 'farbwenz_solist', 'farbwenz_sologegner', 'geier_solist', 'geier_sologegner', 'wenz_solist', 'wenz_sologegner', 'farbsolo_solist', 'farbsolo_sologegner')
# for i, role in enumerate(roles):
#     df.loc[df.role == role, 'role_id'] = i
# df.to_parquet(os.path.join(data_root, f'{condition}.parquet'))

df_present = os.path.exists(os.path.join(data_root, f'{condition}.parquet'))
if df_present:
    df = pd.read_parquet(os.path.join(data_root, f'{condition}.parquet'), columns=cols, engine='fastparquet')
    df = df[df[var_valid] & (df.tdiff_new_max2 < 900)]
    df['performance'] = df['player_augen'] - df['regquality']
    df['regquality_sologegner_cards_best'] -= df['regquality_sologegner_cards_best'].mean()
    df['regquality_sologegner_cards_best'] /= df['regquality_sologegner_cards_best'].std()
    print(f'N={len(df)}')
    print(f'av time between failure and successes = {df.tdiff_new_max2.mean():.1f} seconds')

regs = [
    'prev_won',
    'C(rule_kurze)',
    'C(role_id)',
    'C(cur_pos)'
]

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

if df_present:
    # print(f"\nCorrelation matrix:\n{df[[dv] + [reg.replace('C(', '').replace(')', '') for reg in regs if reg != var_ranS]].corr()}")
    print(f"\nCorrelation matrix:\n{df[[dv] + [reg for reg in regs if (reg != var_ranS) and not reg.startswith('C(')]].corr()}")

    ff_cor = correct_dv(df, dv, result)
    ff_cor.to_parquet(f"../data/figures/{__file__.split('/')[-1].replace('stats', 'prepare_figure').replace('.py', '_corrected.parquet')}")
    print('\nDependent variable:\n', ff_cor)
    print(f'Effect lme4: {result.iloc[1]['Estimate']:.5f}  ||  Effect correction: {ff_cor.loc['prev_won_cor', 'av'] - ff_cor.loc['prev_lost_cor', 'av'] :.5f}')

print(patsy)
export_regression_table(result, __file__)