import pandas as pd
from coldhand.util.stats_util.expand_predictors import expand_predictors
from pymer4.models import Lmer
from datetime import datetime
from coldhand.figures.util.regression import export_regression_table, prepare_result
import os

df = pd.read_parquet('/home/matteo/Dropbox/schafkopf/data/pduser_paper.parquet', engine='fastparquet')
df['subject'] = df['user_id']
df['data_soloquote_rule6'] *= 100
df['data_soloquote_rule8'] *= 100

df6 = df[df.data_ngames_rule6 > 1000][['subject', 'player_optimality_ratio_rule6', 'data_soloquote_rule6', 'data_score_rule6']].copy().reset_index()
df8 = df[df.data_ngames_rule8 > 1000][['subject', 'player_optimality_ratio_rule8', 'data_soloquote_rule8', 'data_score_rule8']].copy().reset_index()
df6 = df6.rename(columns={c: c.replace('_rule6', '') for c in df6.columns if c.endswith('_rule6')})
df6['rule_kurze'] = 1
df6['player_optimality_ratio_z'] = (df6.player_optimality_ratio - df6.player_optimality_ratio.mean()) / df6.player_optimality_ratio.std()
df8 = df8.rename(columns={c: c.replace('_rule8', '') for c in df8.columns if c.endswith('_rule8')})
df8['rule_kurze'] = 0
df8['player_optimality_ratio_z'] = (df8.player_optimality_ratio - df8.player_optimality_ratio.mean()) / df8.player_optimality_ratio.std()

df68 = pd.concat([df6, df8], ignore_index=True)

df68['data_soloquote'] -= df68.data_soloquote.mean()
df68['player_optimality_ratio_z'] -= df68.player_optimality_ratio_z.mean()

print(df68[['data_score', 'player_optimality_ratio_z']].corr())

dv = 'data_score'
regs = [
    'data_soloquote*player_optimality_ratio_z',
    'C(rule_kurze)'
]
patsy = f"{dv} ~ {' + '.join(regs)} + (1|subject)"

cols = [dv, 'subject'] + expand_predictors(regs)
model = Lmer(patsy, data=df68[cols])

filename = __file__.split('/')[-1].replace('.py', '.parquet')
recompute = False
if recompute:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Performing Lmer analysis: {patsy}")
    model.fit(summary=False)
    print(model.summary())
    result = prepare_result(model)
    result.attrs['patsy'] = patsy
    result.attrs['nsamples'] = len(df68)
    result.to_parquet(os.path.join('data', filename))
else:
    result = pd.read_parquet(os.path.join('data', filename))

export_regression_table(result, __file__, print_google_doc_size=True, exact_p=True, include_df=True)