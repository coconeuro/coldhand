import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from coldhand.figures.config import colors, errorbar_kw, errorbar_dot_kw, default_fontsize_categorical
from coldhand.figures.util.export import savefig
from coldhand.figures.util.fontsize import set_fontsize

ff = pd.read_parquet(f"../../data/figures/prepare_{__file__.split('/')[-1].replace('.py', '_corrected.parquet')}")

var_lost = 'prev_lost_cor'
var_won = 'prev_won_cor'

plt.figure(figsize=(3, 3))
plt.errorbar(0, ff.loc[var_won].av, yerr=ff.loc[var_won].se, color=colors['green'], **errorbar_kw)
plt.plot(0, ff.loc[var_won].av, color=colors['green'], **errorbar_dot_kw)
plt.errorbar(1, ff.loc[var_lost].av, yerr=ff.loc[var_lost].se, color=colors['red'], **errorbar_kw)
plt.plot(1, ff.loc[var_lost].av, color=colors['red'], **errorbar_dot_kw)
plt.plot([-0.5, 1.5], [0, 0], 'k', lw=0.5)

y_pline = -0.05
text_p = '***' if ff.loc[var_lost].p < 0.001 else ('**' if ff.loc[var_lost].p < 0.009 else (
    '*' if ff.loc[var_lost].p < 0.05 else ('*' if ff.loc[var_lost].p < 0.1 else 'n.s.')))
plt.plot([0, 1], [y_pline, y_pline], '-', lw=0.5, color='#777')
plt.text(0.5, y_pline, text_p, ha='center', va='center', fontsize=14, bbox=dict(fc='w', ec='none', pad=0), color='#777')

plt.xticks([0, 1], ['Post-\nsuccess', 'Post-\nfailure'])
plt.xlim(-0.5, 1.5)
plt.ylim(-0.35, 0.35)
plt.yticks(np.arange(-0.3, 0.31, 0.1))
plt.text(-0.22, 1.055, 'Scope:', transform=plt.gca().transAxes, fontsize=11, fontweight='bold', color='#222',
         clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
plt.text(0.11, 1.055, 'all contracts', transform=plt.gca().transAxes, fontsize=11, color='#222',
         clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
plt.ylabel(f'Yield')
plt.grid(axis='y', color=colors['grid'])
plt.text(0.02, 0.87, 'B', transform=plt.gcf().transFigure, fontsize=22)

set_fontsize(**default_fontsize_categorical)

plt.tight_layout()
savefig(f"../img/{__file__.split('/')[-1].replace('.py', '.png')}")

print(f"Δ = {ff.loc[var_lost].av:.2f} - {ff.loc[var_won].av:.2f} = {ff.loc[var_lost].av - ff.loc[var_won].av:.2f}")