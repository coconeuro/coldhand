import os

import pandas as pd
from timeit import default_timer
import numpy as np

# mode = 'hothand/solist_all'
# mode = 'hothand_fold/solist_solist'
# mode = 'hothand/solist_solist'
mode = 'hothand/solist_allplayed'
# mode = 'hothand_fold/solist_all'

root = '/home/matteo/schafkopf/'
root_hothand = os.path.join(root, f"{mode.split('/')[0]}")

gametype_dict = {0: 'ramsch', 1: 'rufspiel', 2: 'farbwenz', 3: 'geier', 4: 'wenz', 5: 'farbsolo'}

# df = pd.read_parquet(os.path.join(data_root, f'{mode}_os.parquet'), engine='fastparquet')
df = pd.read_parquet(os.path.join(root_hothand, f"{mode.split('/')[1]}.parquet"), engine='fastparquet')
# df = pd.read_parquet(os.path.join(root_hothand, f"{mode.split('/')[1]}_os.parquet"), engine='fastparquet')
print(len(df))
# cnd = ~df.player0_result.isna() & ~df.player0_card.isna() & df.player_optimality_ratio.isna()
cnd = ~df.player0_points_max.isna()

# df['performance'] = df['player_augen'] - df['regquality']
# for i in range(4):
#     df.loc[df.position_solist == i, 'optimality_opponent'] = df[[f'player{j}_optimality_percentile' for j in range(4) if j != i]].mean(axis=1)
# for i in range(4):
#     df.loc[df.position_solist == i, 'optimality'] = df[f'player{i}_optimality_percentile']
#

# df['ncards_per_player'] = df.apply(lambda x: 8 - 2*x['rule_kurze'], axis=1)


for i in range(4):
    t0 = default_timer()
    df.loc[cnd, f'player{i}_points_max_first'] = df[cnd].apply(lambda x: x[f'player{i}_points_max'][0], axis=1)
for i in range(4):
    df.loc[cnd & (df.cur_pos == i), 'player_points_max_first'] = df[cnd & (df.cur_pos == i)][f'player{i}_points_max_first']

# print('Initial Loop')
# for i in range(4):
#     t0 = default_timer()
#     df.loc[cnd, f'player{i}_points_card'] = df[cnd].apply(lambda x: [[int(v[1]) for x_ in x[f'player{i}_result'][j].split(',') if int((v:=x_.split('-'))[0]) == int(x[f'player{i}_card'][j])][0] for j in range(len(x[f'player{i}_result']))], axis=1)
#     df.loc[cnd, f'player{i}_points_max'] = df[cnd].apply(lambda x: [max([int(x_.split('-')[1]) for x_ in res.split(',')]) for res in x[f'player{i}_result']], axis=1)
#     df.loc[cnd & (df.cur_pos == i), 'player_points_card'] = df[cnd & (df.cur_pos == i)][f'player{i}_points_card']
#     df.loc[cnd & (df.cur_pos == i), 'player_points_max'] = df[cnd & (df.cur_pos == i)][f'player{i}_points_max']
#     df.loc[cnd, f'player{i}_optimality_ratio'] = df[cnd].apply(lambda x: np.nanmean([1 if P == 0 else p / P for p, P in zip(x[f'player{i}_points_card'], x[f'player{i}_points_max'])]), axis=1)
#     df.loc[cnd, f'player{i}_optimalities_ratio'] = df[cnd].apply(lambda x: [1 if P == 0 else p / P for p, P in zip(x[f'player{i}_points_card'], x[f'player{i}_points_max'])], axis=1)
#     print(f'\tLoop {i + 1} / {4}: {default_timer() - t0:.1f} secs')
#
# t0 = default_timer()
# df.loc[cnd, 'player_optimalities_ratio'] = df[cnd].apply(lambda x: [1 if P == 0 else p / P for p, P in zip(x['player_points_card'], x['player_points_max'])], axis=1)
# print(f'\tplayer_optimalities_ratio: {default_timer() - t0:.1f} secs')
# t0 = default_timer()
# df.loc[cnd, 'player_optimality_ratio'] = df[cnd].apply(lambda x: np.nanmean([1 if P == 0 else p / P for p, P in zip(x['player_points_card'], x['player_points_max'])]), axis=1)
# print(f'\tplayer_optimality_ratio: {default_timer() - t0:.1f} secs')
# for i in range(4):
#     t0 = default_timer()
#     df.loc[cnd & (df.cur_pos == i), 'opponent_optimality_ratio'] = df[cnd & (df.cur_pos == i)][[f'player{j}_optimality_ratio' for j in range(4) if j != i]].mean(axis=1)
#     print(f'\topponent_optimality_ratio ({i + 1} / 4): {default_timer() - t0:.1f} secs')
#
#
# df.loc[~df.player_role.isna(), 'role'] = df[~df.player_role.isna()].apply(lambda x: f"{gametype_dict[x['game_type']]}_{x['player_role']}", axis=1)
# roles = ('rufspiel_sauspieler', 'rufspiel_mitspieler', 'rufspiel_sauspielgegner', 'farbwenz_solist', 'farbwenz_sologegner', 'geier_solist', 'geier_sologegner', 'wenz_solist', 'wenz_sologegner', 'farbsolo_solist', 'farbsolo_sologegner')
# for i, role in enumerate(roles):
#     df.loc[df.role == role, 'role_id'] = i


# path_save = os.path.join(root_hothand, f"{mode.split('/')[1]}_os.parquet")
path_save = os.path.join(root_hothand, f"{mode.split('/')[1]}.parquet")
df.to_parquet(path_save, engine='pyarrow')
print(f'File saved to {path_save}')




# for i in range(4):
#     df.loc[cnd, f'player{i}_points_card'] = df[cnd].apply(lambda x: [[int(v[1]) for x_ in x[f'player{i}_result'][j].split(',') if int((v:=x_.split('-'))[0]) == int(x[f'player{i}_card'][j])][0] for j in range(len(x[f'player{i}_result']))], axis=1)
#     df.loc[cnd, f'player{i}_points_max'] = df[cnd].apply(lambda x: [max([int(x_.split('-')[1]) for x_ in res.split(',')]) for res in x[f'player{i}_result']], axis=1)
    # df[f'player{i}_points_min'] = df.apply(lambda x: [min([int(x_.split('-')[1]) for x_ in res.split(',')]) for res in x[f'player{i}_result']], axis=1)
    # df[f'player{i}_percentiles'] = df.apply(lambda x: [((noptions := len(splits := res.split(','))) - (np.array([int(x_.split('-')[1]) for x_ in splits]) > x[f'player{i}_points_card'][j]).sum()) / noptions for j, res in enumerate(x[f'player{i}_result'])], axis=1)
    # for beta in np.arange(0.1, 1.51, 0.1):
    #     df[f'player{i}_points_expsum{10*beta:.0f}'] = df.apply(lambda x: [np.sum([np.exp(beta*int(x_.split('-')[1])) for x_ in res.split(',')]) for res in x[f'player{i}_result']], axis=1)
#
# for i in range(4):
    # df.loc[cnd & (df.position_solist == i), 'solist_points_card'] = df[cnd & (df.position_solist == i)][f'player{i}_points_card']
    # df.loc[cnd & (df.position_solist == i), 'solist_points_max'] = df[cnd & (df.position_solist == i)][f'player{i}_points_max']
    # df.loc[cnd & (df.cur_pos == i), 'player_points_card'] = df[cnd & (df.cur_pos == i)][f'player{i}_points_card']
    # df.loc[cnd & (df.cur_pos == i), 'player_points_max'] = df[cnd & (df.cur_pos == i)][f'player{i}_points_max']
    # df.loc[df.position_solist == i, 'solist_points_min'] = df[f'player{i}_points_min']
    # df.loc[df.position_solist == i, 'solist_percentiles'] = df[f'player{i}_percentiles']
    # for beta in np.arange(0.1, 1.51, 0.1):
    #     df.loc[df.position_solist == i, f'solist_points_expsum{10*beta:.0f}'] = df[f'player{i}_points_expsum{10*beta:.0f}']

# for i in range(4):
#     df.loc[cnd, f'player{i}_optimality_ratio'] = df[cnd].apply(lambda x: np.nanmean([1 if P == 0 else p / P for p, P in zip(x[f'player{i}_points_card'], x[f'player{i}_points_max'])]), axis=1)
#     df.loc[cnd, f'player{i}_optimalities_ratio'] = df[cnd].apply(lambda x: [1 if P == 0 else p / P for p, P in zip(x[f'player{i}_points_card'], x[f'player{i}_points_max'])], axis=1)

# for i in range(4):
#     df.loc[cnd & (df.cur_pos == i), 'opponent_optimalities_ratio'] = df[cnd & (df.cur_pos == i)].apply(lambda x: np.nanmean([x[f'player{j}_optimalities_ratio'] for j in range(4) if j != i], axis=0), axis=1)
    # df.loc[df.position_solist == i, 'opponent_optimality'] = df.apply(lambda x: np.nanmean([x[f'player{j}_percentiles'] for j in range(4) if j != i]), axis=1)
    # df.loc[df.position_solist == i, 'opponent_optimalities'] = df.apply(lambda x: np.nanmean([x[f'player{j}_percentiles'] for j in range(4) if j != i], axis=0), axis=1)

# for i in range(4):
#     df.loc[cnd & (df.cur_pos == i), 'opponent_optimality_ratio'] = df[cnd & (df.cur_pos == i)][[f'player{j}_optimality_ratio' for j in range(4) if j != i]].mean(axis=1)

# df.loc[cnd, 'solist_optimalities_ratio'] = df[cnd].apply(lambda x: [1 if P == 0 else p / P for p, P in zip(x['solist_points_card'], x['solist_points_max'])], axis=1)
# df.loc[cnd, 'player_optimalities_ratio'] = df[cnd].apply(lambda x: [1 if P == 0 else p / P for p, P in zip(x['player_points_card'], x['player_points_max'])], axis=1)
# df['solist_optimalities_diff'] = df.apply(lambda x: [p - P for p, P in zip(x['solist_points_card'], x['solist_points_max'])], axis=1)

# df['solist_optimalities_ratio_rankweighted'] = df.apply(lambda x: [w*r for w, r in zip(x['solist_percentiles'], x['solist_optimalities_ratio'])], axis=1)
# df['solist_optimality_ratio_rankweighted'] = df.apply(lambda x: np.nanmean(x['solist_optimalities_ratio_rankweighted']), axis=1)
# df['solist_optimality_ratio_rankweighted'] = df.apply(lambda x: np.average(x['solist_optimalities_ratio'], weights=x['solist_percentiles']), axis=1)

# df['ausspiel_weights'] = df.apply(lambda x: [(8-2*x['rule_kurze']-i)/(8-2*x['rule_kurze']) for i in range(len(x['solist_optimalities_ratio']))], axis=1)
# df['solist_optimality_ratio_ausspielweighted'] = df.apply(lambda x: np.average(x['solist_optimalities_ratio'], weights=x['ausspiel_weights']), axis=1)
# df['opponent_optimality_ratio_ausspielweighted'] = df.apply(lambda x: np.average(x['opponent_optimalities_ratio'], weights=x['ausspiel_weights']), axis=1)
# df['opponent_optimality_ausspielweighted'] = df.apply(lambda x: np.average(x['opponent_optimalities'], weights=x['ausspiel_weights']), axis=1)

# df['solist_percentiles_ausspielweighted'] = df.apply(lambda x: [r*(8-2*x['rule_kurze']-i)/(8-2*x['rule_kurze']) for i, r in enumerate(x['solist_percentiles'])], axis=1)
# df['optimality_ausspielweighted'] = df.apply(lambda x: np.nanmean(x['solist_percentiles_ausspielweighted']), axis=1)
# df['optimality_ausspielweighted'] = df.apply(lambda x: np.average(x['solist_percentiles'], weights=x['ausspiel_weights']), axis=1)
# df['solist_optimality_diff'] = df.apply(lambda x: np.nanmean([p - P for p, P in zip(x['solist_points_card'], x['solist_points_max'])]), axis=1)
# df.loc[cnd, 'solist_optimality_ratio'] = df[cnd].apply(lambda x: np.nanmean([1 if P == 0 else p / P for p, P in zip(x['solist_points_card'], x['solist_points_max'])]), axis=1)
# df.loc[cnd, 'player_optimality_ratio'] = df[cnd].apply(lambda x: np.nanmean([1 if P == 0 else p / P for p, P in zip(x['player_points_card'], x['player_points_max'])]), axis=1)
# df['solist_optimality_ratio_nan'] = df.apply(lambda x: np.nanmean([np.nan if P == 0 else p / P for p, P in zip(x['solist_points_card'], x['solist_points_max'])]), axis=1)
# df['solist_optimality_ratio_firsthalf'] = df.apply(lambda x: np.nanmean([1 if P == 0 else p / P for p, P in zip(x['solist_points_card'][:4-x['rule_kurze']], x['solist_points_max'][:4-x['rule_kurze']])]), axis=1)
# df['optimality_firsthalf'] = df.apply(lambda x: np.nanmean(x['solist_percentiles'][:4-x['rule_kurze']]), axis=1)
# df['solist_optimality_minmax'] = df.apply(lambda x: np.nanmean([1 if pmax == pmin else (p - pmin) / (pmax - pmin) for p, pmin, pmax in zip(x['solist_points_card'], x['solist_points_min'], x['solist_points_max'])]), axis=1)
# for beta in np.arange(0.1, 1.51, 0.1):
#     df[f'solist_optimality_softmax{10*beta:.0f}'] = df.apply(lambda x: np.nanmean([np.nan if psum == 0 else np.exp(beta*p) / psum for p, psum in zip(x['solist_points_card'], x[f'solist_points_expsum{10*beta:.0f}'])]), axis=1)

# print(df[['optimality', 'player_augen']].corr())
