def expand_predictors(predictors):
    """ From a list with predictors (including interaction terms and categorical predictors)
        extract the list of unique predictor strings.
    """
    return list(set([p_.replace('C(', '').replace(')', '') for p_ in sum([p.split(':') if ':' in p else p.split('*') for p in predictors], [])]))
