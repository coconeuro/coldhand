import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from coldhand.figures.config import colors, errorbar_kw, errorbar_dot_kw, default_fontsize_categorical
from coldhand.figures.util.export import savefig
from coldhand.figures.util.fontsize import set_fontsize
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

root = '../../data/figures/'

# analyses = ('solist_all_yield', 'solist_all_soloquote_linear', 'solist_allplayed_optimality', 'solist_solist_points', 'solist_solist_cardquality', 'solist_solist_optimality')
# ylabels = ('Yield', 'p(Solo) [%]', 'Optimality [%]', 'Card points | Solo     ', 'Cardquality | Solo     ', 'Optimality [%] | Solo     ')
analyses = ('solist_solist_fold_points', 'solist_solist_fold_cardquality', 'solist_solist_fold_optimality', 'solist_all_fold_yield', 'solist_all_fold_soloquote_linear', 'solist_allplayed_fold_optimality')
ylabels = ('Card points | Solo     ', 'Card quality | Solo     ', 'Optimality [%] | Solo   ', 'Yield', 'p(Solo) [%]', 'Optimality [%]')

categories = ('Outcome', 'Risk-Taking', 'Performance')

ylims = (
    (73.25, 74.05),
    # (70.6401, 71.8),
    (70.15, 72.15),
    (97.3501, 97.455),
    (-0.4, 0.48),
    (11.42, 12.5),
    (97.845, 97.925)
)

yticks = (
    np.arange(73.4, 74.01, 0.2),
    # np.arange(70.8, 72, 0.2),
    np.arange(70.5, 72.1, 0.5),
    np.arange(97.37, 97.461, 0.02),
    np.arange(-0.4, 0.41, 0.2),
    np.arange(11.6, 12.6, 0.2),
    np.arange(97.86, 97.925, 0.02)
)

var_lost = 'prev_lost_cor'
var_won = 'prev_won_cor'
var_fold = 'prev_fold_cor'

fig = plt.figure(figsize=(9, 5))
for i, analysis in enumerate(analyses):

    # ff = pd.read_parquet(os.path.join(root, f'prepare_figure_{analysis}_corrected.parquet'))
    ff = pd.read_parquet(os.path.join(root, f'tableS{'ABCDEF'.index('ACEBDF'[i]) + 7}_{analysis}.parquet'))

    # if analysis == 'solist_all_fold_soloquote_linear':
    #     ff[['av', 'se']] *= 100

    ax = plt.subplot(2, 3, i + 1)
    plt.errorbar(0, ff.loc[var_won].av, yerr=ff.loc[var_won].se, color=colors['green'], **errorbar_kw)
    plt.plot(0, ff.loc[var_won].av, color=colors['green'], **errorbar_dot_kw)
    plt.errorbar(1, ff.loc[var_fold].av, yerr=ff.loc[var_fold].se, color=colors['grey'], **errorbar_kw)
    plt.plot(1, ff.loc[var_fold].av, color=colors['grey'], **errorbar_dot_kw)
    plt.errorbar(2, ff.loc[var_lost].av, yerr=ff.loc[var_lost].se, color=colors['red'], **errorbar_kw)
    plt.plot(2, ff.loc[var_lost].av, color=colors['red'], **errorbar_dot_kw)

    # y_pline = (ff.loc[var_won].av + (ff.loc[var_lost].av - ff.loc[var_won].av) / 2)
    # text_p = '***' if ff.loc[var_lost].p < 0.001 else ('**' if ff.loc[var_lost].p < 0.009 else (
    #     '*' if ff.loc[var_lost].p < 0.05 else ('*' if ff.loc[var_lost].p < 0.1 else 'n.s.')))
    # if analysis == 'solist_all_fold_yield':
    #     plt.plot([-0.5, 2.5], [0, 0], 'k', lw=0.5)
    #     y_pline -= 0.05
    # plt.plot([0, 1], [y_pline, y_pline], '-', lw=0.5, color='#777')
    # plt.text(0.5, y_pline, text_p, ha='center', va='center', fontsize=14, bbox=dict(fc='w', ec='none', pad=0), color='#777')

    if analysis == 'solist_all_fold_yield':
        plt.plot([-0.5, 2.5], [0, 0], 'k', lw=0.5)
        # y_pline -= 0.05

    y_pline = (ff.loc[var_fold].av - (ff.loc[var_fold].av - ff.loc[var_won].av) / 2)
    text_p = '***' if ff.loc[var_won].p < 0.001 else ('**' if ff.loc[var_won].p < 0.01 else (
        '*' if ff.loc[var_won].p < 0.05 else ('o' if ff.loc[var_won].p < 0.1 else 'ns')))
    plt.plot([0.13 if text_p == 'ns' else 0.1, 0.9 if text_p == 'ns' else 0.87], [y_pline, y_pline], '-', lw=0.5, color='#777')
    plt.text(0.5, y_pline, text_p, ha='center', va='center', fontsize=14, bbox=dict(fc='w', ec='none', pad=1 if text_p == 'ns' else 0), color='#777')

    y_pline = (ff.loc[var_fold].av - (ff.loc[var_fold].av - ff.loc[var_lost].av) / 2)
    text_p = '***' if ff.loc[var_lost].p < 0.001 else ('**' if ff.loc[var_lost].p < 0.01 else (
        '*' if ff.loc[var_lost].p < 0.05 else ('*' if ff.loc[var_lost].p < 0.1 else 'ns')))
    plt.plot([1.13 if text_p == 'ns' else 1.1, 1.9 if text_p == 'ns' else 1.87], [y_pline, y_pline], '-', lw=0.5, color='#777')
    plt.text(1.5, y_pline, text_p, ha='center', va='center', fontsize=14, bbox=dict(fc='w', ec='none', pad=1 if text_p == 'ns' else 0), color='#777')


    if i > 2:
        # plt.xticks([0, 1, 2], ['    Post-\nsuccess', 'Post-\nneutral', 'Post-   \nfailure'])
        plt.xticks([0, 1, 2], ['', '', ''])
        plt.text(0.25, -0.23, 'Post-\nsuccess', transform=ax.transAxes, clip_on=False, fontsize=13, ha='right', linespacing=0.9)
        plt.text(0.5, -0.23, 'Post-\nneutral', transform=ax.transAxes, clip_on=False, fontsize=13, ha='center', linespacing=0.9)
        plt.text(0.75, -0.23, 'Post-\nfailure', transform=ax.transAxes, clip_on=False, fontsize=13, ha='left', linespacing=0.9)
    else:
        plt.xticks([0, 1, 2], ['', '', ''])
    plt.xlim(-0.5, 2.5)

    if (0 < ylims[i][0]) or (0 > ylims[i][1]):
        yrange = ylims[i][1] - ylims[i][0]
        xmin, xmax, xrange = -0.5, 2.5, 3
        gapwidth = 0.02*xrange
        # Poor man's broken y-axis (leave for future reference)
        gapy, gapheight, gaptilt = ylims[i][0] + 0.08*yrange, 0.02*yrange, 0.005*yrange
        plt.barh(gapy, xrange+2*gapwidth, left=xmin-0.01*xrange, height=gapheight, clip_on=False, lw=0, fc='w', zorder=10)
        for x in (xmin, xmax):
            plt.plot([x-gapwidth, x+gapwidth], [gapy-gapheight/2-gaptilt, gapy-gapheight/2+gaptilt], 'k-', clip_on=False, lw=0.75, zorder=10)
            plt.plot([x-gapwidth, x+gapwidth], [gapy+gapheight/2-gaptilt, gapy+gapheight/2+gaptilt], 'k-', clip_on=False, lw=0.75, zorder=10)
        if yticks[i] is not None:
            plt.yticks(np.hstack((ylims[i][0], yticks[i])), ['0'] + [f'{v:.2f}' if 'optimality' in analysis else f'{v:.1f}' for v in yticks[i]])
    else:
        if yticks[i] is not None:
            plt.yticks(yticks[i])
    plt.ylim(ylims[i])


    plt.ylabel(ylabels[i])
    plt.grid(axis='y', color=colors['grid'])
    plt.title('Unconditional (any contract)      ' if i > 2 else 'Conditional (solo attempts)      ', fontstyle='italic', pad=4)

    if i < 3:
        plt.text((0.01, 0.3475, 0.6775)[i], 0.95, categories[i], transform=fig.transFigure, fontsize=14, fontweight='bold', clip_on=False)

    # plt.text((0.01, 0.33, 0.655)[int(i/2)], (0.87, 0.42)[int(i % 2 == 1)], 'ABCDEF'[i], transform=fig.transFigure, fontsize=19, clip_on=False)
    plt.text(-0.475-0.11*(i in (2, 5)), 0.98, 'ACEBDF'[i], transform=ax.transAxes, fontsize=19, clip_on=False)

    # plt.gca().get_xticklabels()[0].set_transform(plt.gca().get_xticklabels()[0].get_transform() +
    #                                              ScaledTranslation(-1.40, 0, plt.gcf().dpi_scale_trans))
    # plt.gca().get_xticklabels()[1].set_transform(plt.gca().get_xticklabels()[1].get_transform() +
    #                                              ScaledTranslation(-0.02, 0, plt.gcf().dpi_scale_trans))
    # plt.gca().get_xticklabels()[2].set_transform(plt.gca().get_xticklabels()[2].get_transform() +
    #                                              ScaledTranslation(3.40, 0, plt.gcf().dpi_scale_trans))

    set_fontsize(**{**default_fontsize_categorical, **dict(title=11, ylabel=13, xtick=13)})
    for label in ax.get_xticklabels():
        label.set_linespacing(1)
    ax.tick_params(axis='x', which='major', pad=1)

for x in [0.3225, 0.6575]:
    fig.add_artist(Line2D([x, x], [0.05, 0.975], transform=fig.transFigure, color=(0.3, 0.3, 0.3), linewidth=1.25))
# plt.tight_layout()
plt.subplots_adjust(wspace=0.8, hspace=0.3, left=0.1, right=0.97)
savefig(f"../img/{__file__.split('/')[-1].replace('.py', '.png')}")
