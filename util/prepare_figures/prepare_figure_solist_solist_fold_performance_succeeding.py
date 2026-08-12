import os
import numpy as np
import pandas as pd
from scipy.stats import t
from coco_hothand.config import data_root_fold_succeeding
from coco_hothand.util.stats_util.weighted_se import weighted_se

condition = 'solist_solist'
var_valid = 'valid_new_max3'
dv = 'performance'
cols = ['subject', 'player_augen', 'regquality', var_valid, 'prev_won', 'prev_lost', 'prev_fold', f'tdiff_{var_valid[6:]}', 'rule_kurze', 'game_type', 'cur_pos']
df = pd.read_parquet(os.path.join(data_root_fold_succeeding, f'{condition}.parquet'), columns=cols, engine='fastparquet')
df = df[df[var_valid] & (df.tdiff_new_max3 < 900)]
df['performance'] = df['player_augen'] - df['regquality']
n_df = len(df)
print(f'N={n_df}')

m_gb_prevwon, m_gb_prevlost, m_gb_prevfold = df[df.prev_won].groupby('subject')[dv], df[df.prev_lost].groupby('subject')[dv], df[df.prev_fold].groupby('subject')[dv]
m_prevwon_m, m_prevwon_c = m_gb_prevwon.mean().values, m_gb_prevwon.count().values
m_prevwon_m, m_prevwon_c = m_prevwon_m[~pd.isnull(m_prevwon_m)], m_prevwon_c[~pd.isnull(m_prevwon_m)]
m_prevlost_m, m_prevlost_c = m_gb_prevlost.mean().values, m_gb_prevlost.count().values
m_prevlost_m, m_prevlost_c = m_prevlost_m[~pd.isnull(m_prevlost_m)], m_prevlost_c[~pd.isnull(m_prevlost_m)]
m_prevfold_m, m_prevfold_c = m_gb_prevfold.mean().values, m_gb_prevfold.count().values
m_prevfold_m, m_prevfold_c = m_prevfold_m[~pd.isnull(m_prevfold_m)], m_prevfold_c[~pd.isnull(m_prevfold_m)]

m_prevlost, m_prevwon, m_prevfold = np.average(m_prevlost_m, weights=m_prevlost_c), np.average(m_prevwon_m, weights=m_prevwon_c), np.average(m_prevfold_m, weights=m_prevfold_c)

m_diff_lost = m_prevlost - m_prevfold
se_diff_lost = weighted_se(m_prevlost_m - m_prevfold_m, weights=m_prevlost_c)
m_diff_won = m_prevwon - m_prevfold
se_diff_won = weighted_se(m_prevwon_m - m_prevfold_m, weights=m_prevwon_c)
m_diff = m_prevwon - m_prevlost
se_diff = weighted_se(m_prevwon_m - m_prevlost_m, weights=m_prevwon_c)

p_diff_lost = 2 * (1 - t.cdf(np.abs(m_diff_lost / se_diff_lost), len(m_prevlost_m)-1))  # two-tailed test
p_diff_won = 2 * (1 - t.cdf(np.abs(m_diff_won / se_diff_won), len(m_prevwon_m)-1))  # two-tailed test
p_diff = 2 * (1 - t.cdf(np.abs(m_diff / se_diff), len(m_prevwon_m)-1))  # two-tailed test

ff = pd.DataFrame(columns=['av', 'se', 'p'], index=['prev_lost', 'prev_won', 'prev_fold'], dtype=float)

ff.loc['prev_lost'] = [np.average(m_prevlost_m, weights=m_prevlost_c), weighted_se(m_prevlost_m, weights=m_prevlost_c), p_diff_lost]
ff.loc['prev_won'] = [np.average(m_prevwon_m, weights=m_prevwon_c), weighted_se(m_prevwon_m, weights=m_prevwon_c), p_diff_won]
ff.loc['prev_fold'] = [np.average(m_prevfold_m, weights=m_prevfold_c), weighted_se(m_prevfold_m, weights=m_prevfold_c), np.nan]
print(f'[{dv}] prev-lost: {ff.loc['prev_lost']['av']:.5f} +- {ff.loc['prev_lost']['se']:.5f}, prev-won: {ff.loc['prev_won']['av']:.5f} +- {ff.loc['prev_won']['se']:.5f}, prev-lost: {ff.loc['prev_lost']['av']:.5f} +- {ff.loc['prev_lost']['se']:.5f}')
print(ff)
ff.to_parquet(f"../../data/figures/{__file__.split('/')[-1].replace('.py', '.parquet')}")

