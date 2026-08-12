
mean_z_transform_player = 0.97386
std_z_transform_player = 0.03920
mean_z_transform_opponent = 0.98097
std_z_transform_opponent = 0.01822

mean_z_transform_player_all = 0.97929
std_z_transform_player_all = 0.03574
mean_z_transform_opponent_all = 0.97927
std_z_transform_opponent_all = 0.02111

def compute_z_transform(condition='solist_solist_os'):
    import os
    import pandas as pd
    from coco_hothand.config import data_root
    cols = ['player_user_id', 'player_optimality_ratio', 'opponent_optimality_ratio', 'player0_optimality_ratio', 'player1_optimality_ratio', 'player2_optimality_ratio', 'player3_optimality_ratio']
    df = pd.read_parquet(os.path.join(data_root, f'{condition}.parquet'), columns=cols, engine='fastparquet')

    if '_all' in condition:
        print(f'mean_z_transform_player_all = {df.player_optimality_ratio.mean():.5f}')
        print(f'std_z_transform_player_all = {df.player_optimality_ratio.std():.5f}')
        print(f'mean_z_transform_opponent_all = {df.opponent_optimality_ratio.mean():.5f}')
        print(f'std_z_transform_opponent_all = {df.opponent_optimality_ratio.std():.5f}')
    else:
        print(f'mean_z_transform_player = {df.player_optimality_ratio.mean():.5f}')
        print(f'std_z_transform_player = {df.player_optimality_ratio.std():.5f}')
        print(f'mean_z_transform_opponent = {df.opponent_optimality_ratio.mean():.5f}')
        print(f'std_z_transform_opponent = {df.opponent_optimality_ratio.std():.5f}')


if __name__ == '__main__':
    # compute_z_transform('solist_solist_os')
    compute_z_transform('solist_all_os')