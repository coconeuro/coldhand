import os
import dataframe_image as dfimg
from coco_hothand.figures.util.stack_images import stack_images
from coco_hothand.figures.config import predictor_map
from coco_hothand.util.stats_util.print_ranefs import print_ranefs
import pandas as pd
import numpy as np
import rpy2
from PIL import Image
from itertools import combinations

def get_standardizer(df, dv, predictors):
    """

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


def format_regression_table(result, predictor_map=None, standardizer=None, reorder_predictors=None):
    """

    Args:
        result (pd.DataFrame): unformated Pandas dataframe containing the regression table
        predictor_map (dict[str, str], optional): renaming dictionary for the predictors
        standardizer (dict[str, float], optional): standardizer dictionary for the predictors
        reorder_predictors (dict[int, int], optional): position reorder dictionary for the predictors

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
    if standardizer is None:
        del cols['beta']
    else:
        result_fmt.insert(3, 'beta', [f'{result_fmt.Estimate.iloc[i] / standardizer[p]:.3f}' if p in standardizer else '' for i, p in enumerate(predictors)])

    result_fmt = result_fmt[[''] + list(cols.keys())].rename(columns=cols)

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

def export_regression_table(result, filename, ci95_as_percent=False, df=None, reorder_predictors=None, standardizer=None):

    rdf = print_ranefs(result)
    if ci95_as_percent:
        print(f"\nConfidence interval for the difference:\nCI95 = [{100*result.iloc[1]['2.5_ci']:.3f}%; {100*result.iloc[1]['97.5_ci']:.3f}%]")
        print(f"\nConfidence interval for the difference:\nCI95 = [{result.iloc[1]['2.5_ci']:.4f}; {result.iloc[1]['97.5_ci']:.4f}] {result.iloc[1]['Estimate']:.4f} ± {np.abs(result.iloc[1]['Estimate'] - result.iloc[1]['97.5_ci']):.4f}")
        print(f"CI95(odds) = [{100*(np.exp(result.iloc[1]['2.5_ci'])-1):.3f}%; {100*(np.exp(result.iloc[1]['97.5_ci'])-1):.3f}%]")
        # pred_cat = set([k.split(')')[0][2:] for k in result.Estimate.index if k.startswith('C(') and 'prev_cond' not in k])
        # p1 = odds1 / (1 + odds1)
    else:
        if result.index[1].startswith('C(prev_cond)'):
            print(f"\nConfidence interval post-loss versus post-fold:\nCI95 = [{result.iloc[1]['2.5_ci']:.4f}; {result.iloc[1]['97.5_ci']:.4f}] {result.iloc[1]['Estimate']:.4f} ± {np.abs(result.iloc[1]['Estimate'] - result.iloc[1]['97.5_ci']):.4f}")
            print(f"\nConfidence interval post-win versus post-fold:\nCI95 = [{result.iloc[2]['2.5_ci']:.4f}; {result.iloc[2]['97.5_ci']:.4f}] {result.iloc[2]['Estimate']:.4f} ± {np.abs(result.iloc[2]['Estimate'] - result.iloc[2]['97.5_ci']):.4f}")
        else:
            print(f"\nConfidence interval for the difference:\nCI95 = [{result.iloc[1]['2.5_ci']:.4f}; {result.iloc[1]['97.5_ci']:.4f}] {result.iloc[1]['Estimate']:.4f} ± {np.abs(result.iloc[1]['Estimate'] - result.iloc[1]['97.5_ci']):.4f}")
    print(f"\nModel evidence:\nAIC = {result.attrs['AIC']:.5f}")

    # caption_html = """
    #     <div style="font-size:14px; font-style: italic; width: 100%; font-weight: bold; background-color: #f5f5f5; padding: 5px">
    #         Fixed effects
    #     </div>
    # """
    # coefs_f = format_regression_table(result, predictor_map=predictor_map, replacements_extra={'game type': 'solo game type'})
    coefs_f = format_regression_table(result, predictor_map=predictor_map, reorder_predictors=None, standardizer=standardizer)
    print('\nRegression table:\n', coefs_f[coefs_f.columns[1:]])
    # coefs_styler = coefs_f.style.hide(axis="index").format(dict(Estimate='{:.2f}', SE='{:.2f}', t='{:.2f}')).map(lambda x: 'font-weight: bold', subset=['']).set_caption(caption_html)
    coefs_styler = coefs_f.style.hide(axis="index").format(dict(Estimate='{:.2f}', SE='{:.2f}', t='{:.2f}')).map(lambda x: 'font-weight: bold', subset=[''])
    path_fe = f"../figures/img/{filename.split('/')[-1].replace('.py', '.png')}"
    # path_re = f"../figures/img/{filename.split('/')[-1].replace('.py', '_re.png')}"
    print(f'Figure saved to {os.path.abspath(path_fe)}')
    dfimg.export(coefs_styler, path_fe)
    with Image.open(path_fe) as img:
        print(f'Size: {img.width}px x {img.height}px = {2*img.width/100}cm x {2*img.height/100}cm')
    # dfimg.export(rdf, path_re)
    # stacked_img = stack_images(path_fe, path_re)
    # stacked_img.save(f"../figures/img/{filename.split('/')[-1].replace('.py', '_stacked.png')}")
    # os.remove(path_fe)
    # os.remove(path_re)

def prepare_result(model):
    result = model.coefs
    # result.attrs['ranef'] = {'intercept': model.ranef['(Intercept)'].tolist(), var_ranS: model.ranef[f'{var_ranS}TRUE'].tolist()}
    result.attrs['ranef'] = {k: model.ranef[k].tolist() for k in model.ranef.columns}
    result.attrs['ranef_var'] = {k: v for k, v in zip(model.ranef_var['Name'], model.ranef_var['Std']) if k != ''}
    result.attrs['ranef_corr'] = {f'{iv1} versus {iv2}': corr for iv1, iv2, corr in zip(model.ranef_corr['IV1'], model.ranef_corr['IV2'], model.ranef_corr['Corr'])}
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