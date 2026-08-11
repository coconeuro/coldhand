
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
    'prev_wonTRUE': 'PriorOutcome [Success]',
    'C(prev_cond)Lost': 'PriorOutcome [Failure]',
    'C(prev_cond)Won': 'PriorOutcome [Success]',
    'C(rule_kurze)TRUE': 'DeckType [Short]',
    'C(rule_kurze)1': 'DeckType [Short]',
    'regquality': 'CardQuality',
    'regquality_sologegner_cards_best': 'CardQualityBestOpponent',
    'player_optimality_ratio_z': 'PlayerStrength',
    'data_soloquote': 'SoloFrequency',
    # 'C(game_type)1': 'Contract [Rufspiel]',
    'C(game_type)2': 'SoloContract [Farbwenz]',
    'C(game_type)3': 'SoloContract [Geier]',
    'C(game_type)4': 'SoloContract [Wenz]',
    'C(game_type)5': 'SoloContract [Farbsolo]',
    'C(cur_pos)1': 'Position [2]',
    'C(cur_pos)2': 'Position [3]',
    'C(cur_pos)3': 'Position [4]',
    'C(role_id)1': 'ContractRole [PartnerTeammate]',
    'C(role_id)2': 'ContractRole [PartnerOpponent]',
    'C(role_id)3': 'ContractRole [FarbwenzDeclarer]',
    'C(role_id)4': 'ContractRole [FarbwenzOpponent]',
    'C(role_id)5': 'ContractRole [GeierDeclarer]',
    'C(role_id)6': 'ContractRole [GeierOpponent]',
    'C(role_id)7': 'ContractRole [WenzDeclarer]',
    'C(role_id)8': 'ContractRole [WenzOpponent]',
    'C(role_id)9': 'ContractRole [FarbsoloDeclarer]',
    'C(role_id)10': 'ContractRole [FarbsoloOpponent]'
}
