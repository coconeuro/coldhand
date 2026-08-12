import dataframe_image as dfimg
import numpy as np
import os
import pandas as pd
import rpy2
from PIL import Image
from itertools import combinations

from coldhand.figures.config import predictor_map


def get_standardizer(df, dv, predictors):
    """
    Compute standardizer to compute standardized regression coefficients

    Args:
        df (pd.DataFrame): data
        dv (str): name of dependent variable (y)
        predictors (list/tuple of str): predictors

    Returns:
        standardizer (dict): standardizer for each predictor

    """

    # clean categorical predictors
    predictors = [p.replace('C(', '').replace(')', '') for p in predictors]

    standardizer = {}
    for pred in predictors:
        if '*' in pred:
            combos = [list(c) for r in range(1, len(pred)+1) for c in combinations(pred.split('*'), r)]
            for combo in combos:
                standardizer[':'.join(combo)] = df[dv].std() / np.prod([df[p].std() for p in combo])
        elif ':' in pred:
            standardizer[pred] = df[dv].std() / np.prod([df[p].std() for p in pred.split(':')])
        else:
            standardizer[pred] = df[dv].std() / df[pred].std()

    return standardizer


def format_regression_table(result, predictor_map=None, standardizer=None, reorder_predictors=None, exact_p=False, include_df=False):
    """
    Format regression table

    Args:
        result (pd.DataFrame): unformated Pandas dataframe containing the regression table
        predictor_map (dict[str, str], optional): renaming dictionary for the predictors
        standardizer (dict[str, float], optional): standardizer dictionary for the predictors
        reorder_predictors (dict[int, int], optional): position reorder dictionary for the predictors
        exact_p (bool, optional): set True to get p-values in scientific notation
        include_df (bool, optional): set True to include degrees of freedom

    Returns:
        result_fm (pd.DataFrame): formated Pandas dataframe containing the regression table
    """
    result_fmt = result.copy(deep='all')
    predictors_old = result_fmt.index.values
    predictors = predictors_old.copy()

    if predictor_map is not None:
        for i, pred in enumerate(predictors_old):
            for pred_exp in (pred.split(':') if ':' in pred else pred.split('*')):
                if pred_exp in predictor_map.keys():
                    predictors[i] = predictors[i].replace(pred_exp, predictor_map[pred_exp])

    if standardizer is not None:
        for pred_old, pred_new in zip(predictors_old, predictors):
            if pred_old in standardizer:
                standardizer[pred_new] = standardizer[pred_old]

    result_fmt.insert(0, '', predictors)
    cols = {
        'Estimate': 'b',
        'SE': 'SE',
        'beta': 'β',
        ('T' if 'T-stat' in result_fmt.columns else 'Z') + '-stat': ('t' if 'T-stat' in result_fmt.columns else 'z'),
        'P-val': 'p',
        'Sig': 'Sign.'
    }
    if include_df:
        items = list(cols.items())
        items.insert(3, ('DF', 'DF'))
        cols = dict(items)

    if standardizer is None:
        del cols['beta']
    else:
        result_fmt.insert(3, 'beta', [f'{result_fmt.Estimate.iloc[i] / standardizer[p]:.3f}' if p in standardizer else '' for i, p in enumerate(predictors)])

    result_fmt = result_fmt[[''] + list(cols.keys())].rename(columns=cols)

    if exact_p:
        result_fmt.p = [f'{p:.1e}' if p < 0.001 else f'{p:.3f}' for p in result_fmt.p]
    else:
        result_fmt.p = ['< .001' if p < 0.001 else f'{p:.3f}'[1:] for p in result_fmt.p]


    if reorder_predictors is not None:
        order_pred = list(np.arange(len(result_fmt)))
        items_to_move = [(old, reorder_predictors[old], order_pred[old]) for old in sorted(reorder_predictors)]
        for old, _, _ in sorted(items_to_move, reverse=True):
            order_pred.pop(old)
        for i, (_, new, value) in enumerate(items_to_move):
            order_pred.insert(new, value)
        result_fmt = result_fmt.iloc[order_pred].copy()

    return result_fmt

def export_regression_table(result, filename, omit_intercept_stats=True, ci95_as_percent=False, reorder_predictors=None, standardizer=None, print_google_doc_size=False, exact_p=False, include_df=False):

    if ci95_as_percent:
        print(f"\nConfidence interval for the difference:\nCI95 = [{100*result.iloc[1]['2.5_ci']:.3f}%; {100*result.iloc[1]['97.5_ci']:.3f}%]")
        print(f"\nConfidence interval for the difference:\nCI95 = [{result.iloc[1]['2.5_ci']:.4f}; {result.iloc[1]['97.5_ci']:.4f}] "
              f"{result.iloc[1]['Estimate']:.4f} ± {np.abs(result.iloc[1]['Estimate'] - result.iloc[1]['97.5_ci']):.4f}")
        print(f"CI95(odds) = [{100*(np.exp(result.iloc[1]['2.5_ci'])-1):.3f}%; {100*(np.exp(result.iloc[1]['97.5_ci'])-1):.3f}%]")
    else:
        if result.index[1].startswith('C(prev_cond)'):
            print(f"\nConfidence interval post-loss versus post-fold:\nCI95 = [{result.iloc[1]['2.5_ci']:.4f}; {result.iloc[1]['97.5_ci']:.4f}] "
                  f"{result.iloc[1]['Estimate']:.4f} ± {np.abs(result.iloc[1]['Estimate'] - result.iloc[1]['97.5_ci']):.4f}")
            print(f"\nConfidence interval post-win versus post-fold:\nCI95 = [{result.iloc[2]['2.5_ci']:.4f}; {result.iloc[2]['97.5_ci']:.4f}] "
                  f"{result.iloc[2]['Estimate']:.4f} ± {np.abs(result.iloc[2]['Estimate'] - result.iloc[2]['97.5_ci']):.4f}")
        else:
            print(f"\nConfidence interval for the difference:\nCI95 = [{result.iloc[1]['2.5_ci']:.4f}; {result.iloc[1]['97.5_ci']:.4f}] "
                  f"{result.iloc[1]['Estimate']:.4f} ± {np.abs(result.iloc[1]['Estimate'] - result.iloc[1]['97.5_ci']):.4f}")
    print(f"\nModel evidence:\nAIC = {result.attrs['AIC']:.5f}")

    coefs_f = format_regression_table(result, predictor_map=predictor_map,
                                      reorder_predictors=reorder_predictors, standardizer=standardizer, exact_p=exact_p, include_df=include_df)
    print('\nRegression table:\n', coefs_f[coefs_f.columns[1:]])
    # if omit_intercept_stats:
    #     for col in ('t', 'p', 'Sign.'):
    #         coefs_f.at['(Intercept)', col] = ''
    fmt_dict = dict(Estimate='{:.3f}', b='{:.3f}', SE='{:.3f}', DF='{:.1f}', t='{:}' if omit_intercept_stats else '{:.1f}')
    coefs_styler = coefs_f.style.hide(axis="index").format(fmt_dict).map(lambda x: 'font-weight: bold', subset=[''])
    path_fe = f"../figures/img/{filename.split('/')[-1].replace('.py', '.png')}"
    print(f'Figure saved to {os.path.abspath(path_fe)}')
    if omit_intercept_stats:
        coefs_styler.data['t'] = coefs_styler.data.apply(lambda x: f"{x['t']:.1f}", axis=1)
        coefs_styler.data = coefs_styler.data.astype(dict(t=object))
        for col in ('t', 'p', 'Sign.'):
            coefs_styler.data.at['(Intercept)', col] = ''
    dfimg.export(coefs_styler, path_fe)

    # For Google doc only
    if print_google_doc_size:
        with Image.open(path_fe) as img:
            print(f'Size: {img.width}px x {img.height}px = {2*img.width/100}cm x {2*img.height/100}cm')

def prepare_result(model):
    result = model.coefs
    # result = model.params
    result.attrs['ranef'] = {k: model.ranef[k].tolist() for k in model.ranef.columns}
    result.attrs['ranef_var'] = {k: v for k, v in zip(model.ranef_var['Name'], model.ranef_var['Std']) if k != ''}
    if model.ranef_corr is not None:
        result.attrs['ranef_corr'] = {f'{iv1} versus {iv2}': corr for iv1, iv2, corr in
                                      zip(model.ranef_corr['IV1'], model.ranef_corr['IV2'], model.ranef_corr['Corr'])}
    result.attrs['var_resid'] = np.var(model.residuals)
    for attr in ('AIC', 'BIC', 'logLike', 'formula'):
        result.attrs[attr] = getattr(model, attr).to_dict() if isinstance(getattr(model, attr), pd.DataFrame) else getattr(model, attr)
    if isinstance(model.warnings, str):
        result.attrs['warnings'] = model.warnings
    elif isinstance(model.warnings, rpy2.robjects.vectors.StrVector):
        result.attrs['warnings'] = model.warnings[0]
    else:  # assume list
        result.attrs['warnings'] = [None] * len(model.warnings)
        for i, warning in enumerate(model.warnings):
            if isinstance(warning, rpy2.robjects.vectors.StrVector):
                result.attrs['warnings'][i] = warning[0]
            else:  # assume string
                result.attrs['warnings'][i] = warning
    return result