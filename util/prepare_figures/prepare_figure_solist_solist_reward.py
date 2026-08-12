import os
import numpy as np
import pandas as pd
from coco_hothand.config import data_root
from coco_hothand.util.stats_util.weighted_se import weighted_se


condition = 'solist_solist'
var_valid = 'valid_new_max2'
dv = 'score'
cols = [
    'subject',  'is_win', dv, var_valid, 'tdiff_new_max2', 'prev_won'
]
df = pd.read_parquet(os.path.join(data_root, f'{condition}.parquet'), columns=cols, engine='fastparquet')
df = df[df[var_valid] & (df.tdiff_new_max2 < 900)]
n_df = len(df)
print(f'N={n_df}')
print(f'Gewinnquote: {100 * df.is_win.mean():.2f}%, Gewinnquote (prevlost): {100 * df[df.prev_won == 0].is_win.mean():.2f}%, Gewinnquote (prevwon): {100 * df[df.prev_won == 1].is_win.mean():.2f}%')

m_gb_prevwon, m_gb_prevlost = df[df.prev_won == 1].groupby('subject')[dv], df[df.prev_won == 0].groupby('subject')[dv]
m_prevwon_m, m_prevwon_c = m_gb_prevwon.mean().values, m_gb_prevwon.count().values
m_prevwon_m, m_prevwon_c = m_prevwon_m[~pd.isnull(m_prevwon_m)], m_prevwon_c[~pd.isnull(m_prevwon_m)]
m_prevlost_m, m_prevlost_c = m_gb_prevlost.mean().values, m_gb_prevlost.count().values
m_prevlost_m, m_prevlost_c = m_prevlost_m[~pd.isnull(m_prevlost_m)], m_prevlost_c[~pd.isnull(m_prevlost_m)]

ff = pd.DataFrame(columns=['av', 'se'], index=['prev_lost', 'prev_won'], dtype=float)

ff.loc['prev_lost'] = [np.average(m_prevlost_m, weights=m_prevlost_c), weighted_se(m_prevlost_m, weights=m_prevlost_c)]
ff.loc['prev_won'] = [np.average(m_prevwon_m, weights=m_prevwon_c), weighted_se(m_prevwon_m, weights=m_prevwon_c)]
print(f'[{dv}] prev-lost: {ff.loc['prev_lost']['av']:.5f} +- {ff.loc['prev_lost']['se']:.5f}, prev-won: {ff.loc['prev_won']['av']:.5f} +- {ff.loc['prev_won']['se']:.5f}')
print(ff)
ff.to_parquet(f"../../data/figures/{__file__.split('/')[-1].replace('.py', '.parquet')}")