import numpy as np
from coldhand.util.stats_util.weighted_se import weighted_se
import pandas as pd
from itertools import product
from scipy.stats import t
from scipy.special import expit

def correct_dv(df, dv, coefs, correct_fe=True, correct_re=True, use_marginal_means=True, is_logit=False,
               consider_covariance=False):

    if is_logit:
        df[f'{dv}_cor'] = -coefs.loc['(Intercept)', 'Estimate']
    else:
        df[f'{dv}_cor'] = df[dv]
    if correct_fe:
        for k, v in zip(coefs.Estimate.index, coefs.Estimate):
            if k not in ['(Intercept)', 'prev_wonTRUE', 'C(prev_cond)Lost', 'C(prev_cond)Won']:
                if k.startswith('C('):  # e.g. C(game_type)
                    factor, level = k[2:].split(')')[0], k.split(')')[1]
                    if level == 'TRUE':
                        df[f'{dv}_cor'] -= v * df[factor]
                    else:
                        df[f'{dv}_cor'] -= v * (df[factor] == int(level))
                else:
                    df[f'{dv}_cor'] -= v * df[k]

    if correct_re:
        for k, v in coefs.attrs['ranef'].items():
            ranef = df['subject'].map(dict(zip(df.subject.unique(), v)))
            if k == '(Intercept)':
                df[f'{dv}_cor'] -= ranef
            elif k.startswith('C('):
                df[f'{dv}_cor'] -= ranef * (df[k[2:].split(')')[0]] == k.split(')')[1])
            elif 'prev_cond' in k:
                df[f'{dv}_cor'] -= ranef * (df['prev_cond'] == k.split('prev_cond')[1])
            else:
                df[f'{dv}_cor'] -= ranef * df[k.replace('TRUE', '')]

    if is_logit:
        df[f'{dv}_cor'] = expit(-df[f'{dv}_cor'])

    if use_marginal_means and not is_logit:
        # Get categorical predictors
        pred_cat = set([k.split(')')[0][2:] for k in coefs.Estimate.index if k.startswith('C(') and 'prev_cond' not in k])
        # Get levels of categorical predictors
        pred_cat_levels_ = [[True if k.split(')')[1] == 'TRUE' else int(k.split(')')[1]) for k in coefs.Estimate.index if pred in k] for pred in pred_cat]
        # pred_cat_levels = [[False]+levels if levels[0] is True else [levels[0]-1]+levels for pred_cat, levels in zip(pred_cat, pred_cat_levels_)]
        pred_cat_levels = [[False] + levels if levels[0] is True else [int(v) if isinstance(v, (int, float, np.integer, np.floating)) else v for v in sorted(df[pred].unique())] for pred, levels in zip(pred_cat, pred_cat_levels_)]

        # Create combinations of categorical regressors
        combos = list(product(*pred_cat_levels))
        # Create pandas filter strings for all combinations
        combo_strings = [' & '.join([f'{pred} == {c}' for pred, c in zip(pred_cat, combo)]) for combo in combos]
        # Unfortunately, np.average takes axis as the second argument in newer Numpy versions, so
        # we need to provide a lambda intermediary
        weighted_avg = lambda values, weights: np.average(values, weights=weights)
        marginal_mean = weighted_avg(*zip(*[[np.average((gb := df[df.eval(combo_string, engine='python')].groupby('subject')[dv]).mean(), weights=gb.count()), len(gb)] for combo_string in combo_strings]))
        df[f'{dv}_cor'] += marginal_mean - df[f'{dv}_cor'].mean()

    if 'prev_won' in coefs.Estimate.index or 'prev_wonTRUE' in coefs.Estimate.index:
        ff_cor = pd.DataFrame(columns=['av', 'se', 't', 'p'], index=['prev_lost', 'prev_won', 'prev_lost_cor', 'prev_won_cor'], dtype=float)
        for suffix in ('', '_cor'):
            levels = sorted(df.prev_won.unique())
            m_gb_prevlost, m_gb_prevwon = df[df.prev_won == levels[0]].groupby('subject')[f'{dv}{suffix}'], df[df.prev_won == levels[1]].groupby('subject')[f'{dv}{suffix}']
            m_prevlost_m, m_prevlost_c = m_gb_prevlost.mean().values, m_gb_prevlost.count().values
            m_prevlost_m, m_prevlost_c = m_prevlost_m[~pd.isnull(m_prevlost_m)], m_prevlost_c[~pd.isnull(m_prevlost_m)]
            m_prevwon_m, m_prevwon_c = m_gb_prevwon.mean().values, m_gb_prevwon.count().values
            m_prevwon_m, m_prevwon_c = m_prevwon_m[~pd.isnull(m_prevwon_m)], m_prevwon_c[~pd.isnull(m_prevwon_m)]

            # Compute p-values
            m_prevlost, m_prevwon = np.average(m_prevlost_m, weights=m_prevlost_c), np.average(m_prevwon_m, weights=m_prevwon_c)
            m_diff = m_prevwon - m_prevlost
            if consider_covariance:
                se_diff = compute_precise_standard_error(df, coefs, pred_cat)
            else:
                se_diff = weighted_se(m_prevwon_m - m_prevlost_m, weights=m_prevwon_c)

            t_diff = m_diff / se_diff
            p_diff = 2 * (1 - t.cdf(np.abs(t_diff), len(m_prevlost_m)-1))  # two-tailed test

            ff_cor.loc[f'prev_lost{suffix}'] = [m_prevlost, weighted_se(m_prevlost_m, weights=m_prevlost_c), t_diff, p_diff]
            ff_cor.loc[f'prev_won{suffix}'] = [m_prevwon, weighted_se(m_prevwon_m, weights=m_prevwon_c), t_diff, p_diff]
    else:
        ff_cor = pd.DataFrame(columns=['av', 'se', 't', 'p'], index=['prev_fold', 'prev_lost', 'prev_won', 'prev_fold_cor', 'prev_lost_cor', 'prev_won_cor'], dtype=float)
        for suffix in ('', '_cor'):
            m_gb_prevfold, m_gb_prevlost, m_gb_prevwon = df[df.prev_cond == 'Fold'].groupby('subject')[f'{dv}{suffix}'], df[df.prev_cond == 'Lost'].groupby('subject')[f'{dv}{suffix}'], df[df.prev_cond == 'Won'].groupby('subject')[f'{dv}{suffix}']
            m_prevfold_m, m_prevfold_c = m_gb_prevfold.mean().values, m_gb_prevfold.count().values
            m_prevfold_m, m_prevfold_c = m_prevfold_m[~pd.isnull(m_prevfold_m)], m_prevfold_c[~pd.isnull(m_prevfold_m)]
            m_prevlost_m, m_prevlost_c = m_gb_prevlost.mean().values, m_gb_prevlost.count().values
            m_prevlost_m, m_prevlost_c = m_prevlost_m[~pd.isnull(m_prevlost_m)], m_prevlost_c[~pd.isnull(m_prevlost_m)]
            m_prevwon_m, m_prevwon_c = m_gb_prevwon.mean().values, m_gb_prevwon.count().values
            m_prevwon_m, m_prevwon_c = m_prevwon_m[~pd.isnull(m_prevwon_m)], m_prevwon_c[~pd.isnull(m_prevwon_m)]

            # Compute p-values
            m_prevfold, m_prevlost, m_prevwon = np.average(m_prevfold_m, weights=m_prevfold_c), np.average(m_prevlost_m, weights=m_prevlost_c), np.average(m_prevwon_m, weights=m_prevwon_c)
            m_diff_lost = m_prevlost - m_prevfold
            se_diff_lost = weighted_se(m_prevlost_m - m_prevfold_m, weights=m_prevlost_c)
            m_diff_won = m_prevwon - m_prevfold
            se_diff_won = weighted_se(m_prevwon_m - m_prevfold_m, weights=m_prevwon_c)
            t_diff_lost, t_diff_won = m_diff_lost / se_diff_lost, m_diff_won / se_diff_won
            p_diff_lost = 2 * (1 - t.cdf(np.abs(t_diff_lost), len(m_prevlost_m)-1))  # two-tailed test
            p_diff_won = 2 * (1 - t.cdf(np.abs(t_diff_won), len(m_prevwon_m)-1))  # two-tailed test

            ff_cor.loc[f'prev_fold{suffix}'] = [m_prevfold, weighted_se(m_prevfold_m, weights=m_prevfold_c), np.nan, np.nan]
            ff_cor.loc[f'prev_lost{suffix}'] = [m_prevlost, weighted_se(m_prevlost_m, weights=m_prevlost_c), t_diff_lost, p_diff_lost]
            ff_cor.loc[f'prev_won{suffix}'] = [m_prevwon, weighted_se(m_prevwon_m, weights=m_prevwon_c), t_diff_won, p_diff_won]

    return ff_cor


def compute_precise_standard_error(df, coefs, pred_cat):
    # The exact computation involves
    # https://stats.stackexchange.com/questions/619497/how-are-the-standard-errors-of-fixed-effects-computed-in-a-mixed-effects-model
    from scipy.sparse import csr_matrix, diags
    from scipy.sparse.linalg import cg, LinearOperator
    # X = Fixed effect design matrix
    X = np.hstack((np.ones((len(df), 1)), np.array(df[['prev_won'] + list(pred_cat)].values, float)))
    n, k = X.shape
    # Z = Random effect design matrix
    Z = csr_matrix(np.ones((n, 2), int))
    # G = covariance matrix of the random effects
    G = np.full((2, 2), np.nan)
    G[0, 0] = coefs.attrs['ranef_var']['(Intercept)']
    G[1, 1] = coefs.attrs['ranef_var']['prev_wonTRUE']
    G[0, 1] = list(coefs.attrs['ranef_corr'].values())[0]
    G[1, 0] = list(coefs.attrs['ranef_corr'].values())[0]
    G = csr_matrix(G)
    # I = residual variance matrix
    I = diags(coefs.attrs['var_resid'] * np.ones(n, int), format='csr')

    # In theory we now need to compute the observation-level variance/covariance matrix V
    # as follows:
    # V = Z @ (G @ Z.T) + I
    # .. and from this compute the covariance matrix of th estimates
    # beta_covariance = np.linalg.inv(X.T @ np.linalg.inv(V) @ X)
    # .. however, this requires too much memory, so with the help of ChatGPT we do it as follows instead:
    V_op = LinearOperator((n, n), matvec=lambda x: Z @ (G @ (Z.T @ x)) + I @ x)
    beta_covariance = np.zeros((k, k))
    # Compute V_inv_X by solving V x = X[:, j] for each column of X
    for j in range(k):
        x = cg(V_op, X[:, j])[0]
        for i in range(j, k):
            beta_covariance[j, i] = np.dot(X[:, i], x)
            if j != i:
                beta_covariance[i, j] = beta_covariance[j, i]

    beta_covariance_inv = np.linalg.inv(beta_covariance)
    standard_errors = np.sqrt(np.diag(beta_covariance_inv))
    se_diff = standard_errors[1]
    return se_diff