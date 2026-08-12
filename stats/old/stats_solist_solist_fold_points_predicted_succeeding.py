import os
import pandas as pd
from coco_hothand.config import data_root_fold_succeeding
from pymer4.models import Lmer
import dataframe_image as dfimg
from coco_hothand.figures.util.regression import format_regression_table
from coco_hothand.figures.config import predictor_map
from datetime import datetime

condition = 'solist_solist'
var_valid = 'valid_new_max3'
dv = 'regquality'
cols = [
    'subject',  dv, var_valid, 'tdiff_new_max3', 'prev_lost', 'prev_won', 'prev_fold',
    'rule_kurze', 'game_type', 'cur_pos', 'spielstaerke'
]
df = pd.read_parquet(os.path.join(data_root_fold_succeeding, f'{condition}.parquet'), columns=cols, engine='fastparquet')
df = df[df[var_valid] & (df.tdiff_new_max3 < 900)]
df.loc[df.prev_fold, 'prev_cond'] = 'Fold'
df.loc[df.prev_lost, 'prev_cond'] = 'Lost'
df.loc[df.prev_won, 'prev_cond'] = 'Won'
n_df = len(df)
print(f'N={n_df}')

reload = True
filename = __file__.split('/')[-1].replace('.py', '.parquet')
if reload:
    regs = [
        'C(prev_cond)',
        'rule_kurze',
        'C(game_type)', 'C(cur_pos)'
    ]

    patsy = f"{dv} ~ {' + '.join(regs)}"
    model = Lmer(f"{patsy} + (1|subject)", data=df[[dv, 'subject'] + [reg.replace('C(', '').replace(')', '') for reg in regs]])

    print(f'[{datetime.now().strftime('%H:%M:%S')}] Performing Lmer analysis: {patsy}')
    model.fit(summary=False)
    print(f'\t[{datetime.now().strftime('%H:%M:%S')}] ... Lmer finished')
    coefs = model.coefs
    coefs.to_parquet(os.path.join('data', filename))
else:
    coefs = pd.read_parquet(os.path.join('data', filename))
print(f"CI95 = [{coefs.iloc[1]['2.5_ci']:.2f}; {coefs.iloc[1]['97.5_ci']:.2f}]")
coefs_f = format_regression_table(coefs, predictor_map=predictor_map)
print(coefs_f[coefs_f.columns[1:]])
coefs_styler = coefs_f.style.hide(axis="index").format(dict(Estimate='{:.2f}', SE='{:.2f}', t='{:.2f}'))
dfimg.export(coefs_styler, f"../figures/img/{__file__.split('/')[-1].replace('.py', '.png')}")