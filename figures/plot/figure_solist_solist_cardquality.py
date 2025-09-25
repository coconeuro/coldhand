import matplotlib.pyplot as plt
from coco_hothand.figures.util.fontsize import set_fontsize
from coco_hothand.figures.util.export import savefig
from coco_hothand.figures.config import colors, errorbar_kw, errorbar_dot_kw, default_fontsize_categorical
import pandas as pd
import numpy as np


ff = pd.read_parquet(f"../../data/figures/prepare_{__file__.split('/')[-1].replace('.py', '_corrected.parquet')}")

var_lost = 'prev_lost_cor'
var_won = 'prev_won_cor'
vals = np.hstack((ff.loc[var_lost].av + np.array([-1, 1]) * ff.loc[var_lost].se, ff.loc[var_won].av + np.array([-1, 1]) * ff.loc[var_won].se))
y_min, y_max = np.min(vals) - 0.1 * (np.max(vals) - np.min(vals)), np.max(vals) + 0.1 * (np.max(vals) - np.min(vals))

plt.figure(figsize=(3.2, 3))
plt.errorbar(0, ff.loc[var_won].av, yerr=ff.loc[var_won].se, color=colors['green'], **errorbar_kw)
plt.plot(0, ff.loc[var_won].av, color=colors['green'], **errorbar_dot_kw)
plt.errorbar(1, ff.loc[var_lost].av, yerr=ff.loc[var_lost].se, color=colors['red'], **errorbar_kw)
plt.plot(1, ff.loc[var_lost].av, color=colors['red'], **errorbar_dot_kw)

y_pline = (ff.loc[var_won].av + (ff.loc[var_lost].av - ff.loc[var_won].av) / 2)
text_p = '***' if ff.loc[var_lost].p < 0.001 else ('**' if ff.loc[var_lost].p < 0.009 else ('*' if ff.loc[var_lost].p < 0.05 else ('*' if ff.loc[var_lost].p < 0.1 else 'n.s.')))
plt.plot([0, 1], [y_pline, y_pline], '-', lw=0.5, color='#777')
plt.text(0.5, y_pline, text_p, ha='center', va='center', fontsize=14, bbox=dict(fc='w', ec='none', pad=0), color='#777')

plt.xticks([0, 1], ['Post-\nsuccess', 'Post-\nfailure'])
plt.xlim(-0.5, 1.5)
# plt.ylim(73.72, 74.38)
# plt.ylim(72.5, 73.47)
# plt.text(-0.22, 1.055, ' Scope:', transform=plt.gca().transAxes, fontsize=10, fontweight='bold', color='#222', clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
# plt.text(0.08, 1.055, 'following soloist games ', transform=plt.gca().transAxes, fontsize=10, color='#222', clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
plt.text(-0.20, 1.055, 'Scope:', transform=plt.gca().transAxes, fontsize=11, fontweight='bold', color='#222', clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
plt.text(0.1, 1.055, 'declared solo contracts', transform=plt.gca().transAxes, fontsize=11, color='#222', clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
# plt.title('Earned points')
# plt.ylabel(f'Card quality', labelpad=19)
# plt.text(0.15, 0.55, '[predicted points]', transform=plt.gcf().transFigure, fontsize=12, ha='center', va='center', rotation=90)
# plt.ylabel(f'Card quality [points]   ')
plt.ylabel(f'Card quality   ', labelpad=17)
plt.text(0.14, 0.55, '[projected points]    ', transform=plt.gcf().transFigure, fontsize=12, ha='center', va='center', rotation=90)
plt.grid(axis='y', color=colors['grid'])
plt.text(0.02, 0.87, 'D', transform=plt.gcf().transFigure, fontsize=22)

# set_fontsize(**{**default_fontsize_categorical, **dict(title=14)})
set_fontsize(**default_fontsize_categorical)

plt.tight_layout()
savefig(f"../img/{__file__.split('/')[-1].replace('.py', '.png')}")

print(f"Δ = {ff.loc[var_lost].av - ff.loc[var_won].av:.2f}")
