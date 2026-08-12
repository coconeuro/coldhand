import os
import pandas as pd
from glob import glob
import numpy as np
from datetime import datetime

root = '/data/pdall_os_kurze/'

files = sorted(glob(os.path.join(root, '*.parquet')))
nfiles = len(files)

cols = sum([[f'player{i}_user_id', f'player{i}_optimality_ratio'] for i in range(4)], [])

data, ndata = dict(), dict()

for i, file in enumerate(files):
    print(f'[{datetime.now().strftime('%H:%M:%S')}] File {i + 1} / {nfiles} [{file}]')
    df = pd.read_parquet(file, columns=cols, engine='fastparquet')
    v = np.vstack(([df[[f'player{i}_user_id', f'player{i}_optimality_ratio']].values for i in range(4)])).astype(float)
    sids = v[:, 0]
    for sid in np.unique(np.unique(sids)).astype(int):
        values = v[sids == sid, 1]
        values_ = values[~np.isnan(values)]
        nvalues = len(values_)
        if nvalues:
            if sid in data:
                ndata_sid = ndata[sid]
                data[sid] = (nvalues * np.mean(values_) + ndata_sid * data[sid]) / (ndata_sid + nvalues)
                ndata[sid] = nvalues + ndata_sid
            else:
                data[sid] = np.mean(values_)
                ndata[sid] = nvalues

d = pd.DataFrame(index=range(len(data)))
d['subject'] = ndata.keys()
d['player_optimality_ratio'] = data.values()
d['ngames'] = ndata.values()
d.sort_values('subject').to_parquet('/home/matteo/schafkopf/player_optimality_ratio.parquet')