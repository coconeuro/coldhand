data_root = '/home/matteo/Dropbox/schafkopf/data/hothand/'
data_root_fold = '/home/matteo/Dropbox/schafkopf/data/hothand_fold/'
# data_root = '/path/to/data/folder/without/foldgames/'
# data_root_fold = '/path/to/data/folder/with/foldgames/'

lme4_optimizers = dict(
    nlopt_bobyqa="optimizer='nloptwrap', optCtrl=list(algorithm='NLOPT_LN_BOBYQA')",
    nlopt_neldermead="optimizer='nloptwrap', optCtrl=list(algorithm='NLOPT_LN_NELDERMEAD')",
    nlminb="optimizer='nlminbwrap'",
    nmkbw="optimizer='nmkbw'",
    bobyqa="optimizer='bobyqa'",
    neldermead="optimizer='Nelder_Mead'",
    lbfgsb="optimizer='optimx', optCtrl=list(method='L-BFGS-B')"
)