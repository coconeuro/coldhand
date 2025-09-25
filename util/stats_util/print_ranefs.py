import numpy as np
import pandas as pd
from scipy.stats import pearsonr

def print_ranefs(result):
    ranef = result.attrs['ranef']
    old_new_map = {
        '(Intercept)': 'intercept',
        'C(prev_cond)Lost': 'preceding failure', 'C(prev_cond)Won': 'preceding success',
        'prev_condLost': 'preceding failure', 'prev_condWon': 'preceding success',
        'prev_wonTRUE': 'preceding success'}
    vars_ranS_new = [old_new_map[k] if k in old_new_map else k for k in ranef.keys()]
    rdf = pd.DataFrame(columns=vars_ranS_new, index=vars_ranS_new, dtype=float)
    rdf2 = pd.DataFrame(columns=vars_ranS_new, index=vars_ranS_new, dtype=float)
    for var_ranS, var_ranS_new in zip(ranef.keys(), vars_ranS_new):
        for var_ranS2, var_ranS_new2 in zip(ranef.keys(), vars_ranS_new):
            if var_ranS_new == var_ranS_new2:
                rdf.loc[var_ranS_new, var_ranS_new2] = np.std(ranef[var_ranS])
                rdf2.loc[var_ranS_new, var_ranS_new2] = result.attrs['ranef_var'][var_ranS]
            else:
                rdf.loc[var_ranS_new, var_ranS_new2] = pearsonr(ranef[var_ranS], ranef[var_ranS2]).statistic
                corr = result.attrs['ranef_corr'][f'{var_ranS} versus {var_ranS2}'] if f'{var_ranS} versus {var_ranS2}' in result.attrs['ranef_corr'] else result.attrs['ranef_corr'][f'{var_ranS2} versus {var_ranS}']
                rdf2.loc[var_ranS_new, var_ranS_new2] = np.nan if corr == '' else corr
    print('\nRandom effect structure (conditional on the data):\n', rdf)
    print('\nRandom effects structure (unconditional / marginal):\n', rdf2)

    caption_html = """
        <div style="font-size:14px; font-style: italic; width: 100%; font-weight: bold; background-color: #f5f5f5; padding: 5px">
            RE structure (standard deviation-correlation matrix)
        </div>
    """
    rdf = rdf.style.format({k: '{:.3f}' for k in vars_ranS_new}).set_caption(caption_html)
    return rdf
