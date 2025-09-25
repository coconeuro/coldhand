
colors = dict(
    red='#a71a01',
    green='#237002',
    grey='#5a5a5a',
    grid='#eee'
)

default_fontsize = dict(label=14, xtick=12, ytick=12, title=14)
default_fontsize_categorical = {**default_fontsize, **dict(xtick=14)}

errorbar_kw = dict(capsize=6, lw=2, capthick=2)
errorbar_dot_kw = dict(marker='o', markersize=4)

predictor_map = {
    '(Intercept)': 'Intercept',
    'prev_wonTRUE': 'PrecedingOutcome [Success]',
    'C(prev_cond)Lost': 'PrecedingOutcome [Failure]',
    'C(prev_cond)Won': 'PrecedingOutcome [Success]',
    'C(rule_kurze)TRUE': 'DeckType [Short]',
    'regquality': 'CardQuality',
    'regquality_sologegner_cards_best': 'CardQualityBestOpponent',
    'C(game_type)1': 'Contract [Rufspiel]',
    'C(game_type)2': 'Contract [Farbwenz]',
    'C(game_type)3': 'Contract [Geier]',
    'C(game_type)4': 'Contract [Wenz]',
    'C(game_type)5': 'Contract [Farbsolo]',
    'C(cur_pos)1': 'Position [2]',
    'C(cur_pos)2': 'Position [3]',
    'C(cur_pos)3': 'Position [4]',
    'C(role_id)1': 'Role [RufspielPartner]',
    'C(role_id)2': 'Role [RufspielOpponent]',
    'C(role_id)3': 'Role [FarbwenzDeclarer]',
    'C(role_id)4': 'Role [FarbwenzOpponent]',
    'C(role_id)5': 'Role [GeierDeclarer]',
    'C(role_id)6': 'Role [GeierOpponent]',
    'C(role_id)7': 'Role [WenzDeclarer]',
    'C(role_id)8': 'Role [WenzOpponent]',
    'C(role_id)9': 'Role [FarbsoloDeclarer]',
    'C(role_id)10': 'Role [FarbsoloOpponent]'
}
