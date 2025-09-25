import matplotlib.pyplot as plt
from coco_hothand.figures.util.fontsize import set_fontsize
from coco_hothand.figures.util.export import savefig
from coco_hothand.figures.config import colors, errorbar_kw, errorbar_dot_kw, default_fontsize_categorical
import pandas as pd
from matplotlib.transforms import ScaledTranslation

ff = pd.read_parquet(f"../../data/figures/prepare_{__file__.split('/')[-1].replace('.py', '_corrected.parquet')}")

var_lost = 'prev_lost_cor'
var_won = 'prev_won_cor'
var_fold = 'prev_fold_cor'

plt.figure(figsize=(3.3, 3))
plt.errorbar(0, ff.loc[var_won].av, yerr=ff.loc[var_won].se, color=colors['green'], **errorbar_kw)
plt.plot(0, ff.loc[var_won].av, color=colors['green'], **errorbar_dot_kw)
plt.errorbar(1, ff.loc[var_fold].av, yerr=ff.loc[var_fold].se, color=colors['grey'], **errorbar_kw)
plt.plot(1, ff.loc[var_fold].av, color=colors['grey'], **errorbar_dot_kw)
plt.errorbar(2, ff.loc[var_lost].av, yerr=ff.loc[var_lost].se, color=colors['red'], **errorbar_kw)
plt.plot(2, ff.loc[var_lost].av, color=colors['red'], **errorbar_dot_kw)
plt.plot([-0.5, 2.5], [0, 0], 'k', lw=0.5, zorder=10)

y_pline = (ff.loc[var_fold].av - (ff.loc[var_fold].av - ff.loc[var_won].av) / 2)
text_p = '***' if ff.loc[var_won].p < 0.001 else ('**' if ff.loc[var_won].p < 0.01 else ('*' if ff.loc[var_won].p < 0.05 else ('o' if ff.loc[var_won].p < 0.1 else 'n.s.')))
plt.plot([0, 1], [y_pline, y_pline], '-', lw=0.5, color='#777')
plt.text(0.5, y_pline, text_p, ha='center', va='center', fontsize=14, bbox=dict(fc='w', ec='none'), color='#777')
y_pline = (ff.loc[var_fold].av + (ff.loc[var_lost].av - ff.loc[var_fold].av) / 2)
text_p = '***' if ff.loc[var_lost].p < 0.001 else ('**' if ff.loc[var_lost].p < 0.01 else ('*' if ff.loc[var_lost].p < 0.05 else ('o' if ff.loc[var_lost].p < 0.1 else 'n.s.')))
plt.plot([1.13, 1.87], [y_pline, y_pline], '-', lw=0.5, color='#777')
plt.text(1.5, y_pline, text_p, ha='center', va='center', fontsize=14, bbox=dict(fc='w', ec='none'), color='#777')

plt.xticks([0, 1, 2], ['Post-\nsuccess', 'Post-\nneutral', 'Post-\nfailure'])
plt.xlim(-0.5, 2.5)
plt.ylim(-0.4, 0.45)
plt.text(-0.09, 1.055, 'Scope:', transform=plt.gca().transAxes, fontsize=10, fontweight='bold', color='#222', clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
plt.text(0.18, 1.055, 'all contracts', transform=plt.gca().transAxes, fontsize=10, color='#222', clip_on=False, bbox=dict(facecolor='#eee', ec='none', pad=3.25))
plt.ylabel(f'Yield')
plt.grid(axis='y', color=colors['grid'])
plt.text(0.02, 0.87, 'B', transform=plt.gcf().transFigure, fontsize=22)

plt.gca().get_xticklabels()[0].set_transform(plt.gca().get_xticklabels()[0].get_transform() + ScaledTranslation(-0.07, 0, plt.gcf().dpi_scale_trans))
plt.gca().get_xticklabels()[1].set_transform(plt.gca().get_xticklabels()[1].get_transform() + ScaledTranslation(-0.02, 0, plt.gcf().dpi_scale_trans))
plt.gca().get_xticklabels()[2].set_transform(plt.gca().get_xticklabels()[2].get_transform() + ScaledTranslation(0.07, 0, plt.gcf().dpi_scale_trans))

set_fontsize(**default_fontsize_categorical)

plt.tight_layout()
savefig(f"../img/{__file__.split('/')[-1].replace('.py', '.png')}")

print(f"Δ (lost vs. fold) = {ff.loc[var_lost].av - ff.loc[var_fold].av:.2f}")
print(f"Δ (won vs. fold) = {ff.loc[var_won].av - ff.loc[var_fold].av:.2f}")