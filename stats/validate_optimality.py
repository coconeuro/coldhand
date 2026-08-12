import os
import pandas as pd
from coco_hothand.config import data_root
from scipy.stats import pearsonr

condition = 'solist_allplayed'

reload = False
if reload:
    cols = ['subject', 'player_optimality_ratio', 'player_points_max_first', 'score', 'is_win', 'player_augen', 'solist']
    df = pd.read_parquet(os.path.join(data_root, f'{condition}.parquet'), columns=cols, engine='fastparquet')
    df.to_parquet(os.path.join(data_root, f'{condition}_validation.parquet'))
else:
    df = pd.read_parquet(os.path.join(data_root, f'{condition}_validation.parquet'))

cols = ['user_id', 'data_score', 'data_ngames']
d = pd.read_parquet('/home/matteo/schafkopf/pduser.parquet', columns=cols, engine='fastparquet')
d = d[d.user_id.isin(df.subject.unique())]
valid_users = d[d.data_ngames >= 1000].user_id.unique()
data_score = d[d.user_id.isin(valid_users)].data_score.values
optimality = df[df.subject.isin(valid_users)].groupby('subject').player_optimality_ratio.mean().values

op = pd.read_parquet('/home/matteo/schafkopf/player_optimality_ratio.parquet')
op = op[(op.ngames >= 1000) & op.subject.isin(valid_users)]

result = pearsonr(op.player_optimality_ratio, d[d.user_id.isin(op.subject)].data_score)
print(f'[between, all data] optimality ~ score: r={result.statistic:.4f}, p={result.pvalue}')

result = pearsonr(data_score, optimality)
print(f'[between, solist_allplayed] optimality ~ score: r={result.statistic:.3f}, p={result.pvalue}')

print(f'[won]: optimality = {100*df[df.is_win].player_optimality_ratio.mean():.4f} ± {100*df[df.is_win].player_optimality_ratio.sem():.4f}')
print(f'[lost]: optimality = {100*df[~df.is_win].player_optimality_ratio.mean():.4f} ± {100*df[~df.is_win].player_optimality_ratio.sem():.4f}')

result = pearsonr(df[~df.player_optimality_ratio.isna()].player_optimality_ratio, df[~df.player_optimality_ratio.isna()].score)
print(f'optimality ~ score: r={result.statistic:.3f}, p={result.pvalue}')

result = pearsonr(df[~df.player_optimality_ratio.isna()].player_optimality_ratio, df[~df.player_optimality_ratio.isna()].player_augen)
print(f'optimality ~ augen: r={result.statistic:.3f}, p={result.pvalue}')

print(f'[won]: cardquality = {df[df.is_win].player_points_max_first.mean():.4f} ± {df[df.is_win].player_points_max_first.sem():.4f}')
print(f'[lost]: cardquality = {df[~df.is_win].player_points_max_first.mean():.4f} ± {df[~df.is_win].player_points_max_first.sem():.4f}')

print(f'[solist, won]: cardquality = {df[df.is_win & df.solist].player_points_max_first.mean():.4f} ± {df[df.is_win & df.solist].player_points_max_first.sem():.4f}')
print(f'[solist, lost]: cardquality = {df[~df.is_win & df.solist].player_points_max_first.mean():.4f} ± {df[~df.is_win & df.solist].player_points_max_first.sem():.4f}')

result = pearsonr(df[~df.player_points_max_first.isna()].player_points_max_first, df[~df.player_points_max_first.isna()].player_augen)
print(f'cardquality ~ augen: r={result.statistic:.4f}, p={result.pvalue}')

result = pearsonr(df[df.solist][~df[df.solist].player_points_max_first.isna()].player_points_max_first, df[df.solist][~df[df.solist].player_points_max_first.isna()].player_augen)
print(f'[solist] cardquality ~ augen: r={result.statistic:.4f}, p={result.pvalue}')