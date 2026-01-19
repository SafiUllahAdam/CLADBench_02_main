
def generate_AD_dictionnary(learning_mode):
    model_dict = []
    # unsupervised algorithms
    if learning_mode == 'unsupervise':
        from adbench.baseline.PyOD import PYOD
        from adbench.baseline.DAGMM.run import DAGMM

        # from pyod
        for _ in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD', 'FeatureBagging', 'HBOS', 'KNN', 'LODA',
                    'LOF', 'LSCP', 'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL', 'DeepSVDD']:
            model_dict[_] = PYOD

        # DAGMM
        model_dict['DAGMM'] = DAGMM

    # semi-supervised algorithms
    elif learning_mode == 'semi-supervise':
        from adbench.baseline.PyOD import PYOD
        from adbench.baseline.GANomaly.run import GANomaly
        from adbench.baseline.DeepSAD.src.run import DeepSAD
        from adbench.baseline.REPEN.run import REPEN
        from adbench.baseline.DevNet.run import DevNet
        from adbench.baseline.PReNet.run import PReNet
        from adbench.baseline.FEAWAD.run import FEAWAD

        model_dict = {'GANomaly': GANomaly,
                            'DeepSAD': DeepSAD,
                            'REPEN': REPEN,
                            'DevNet': DevNet,
                            'PReNet': PReNet,
                            'FEAWAD': FEAWAD,
                            'XGBOD': PYOD}

    # fully-supervised algorithms
    elif learning_mode == 'supervise':
        from adbench.baseline.Supervised import supervised
        from adbench.baseline.FTTransformer.run import FTTransformer

        # from sklearn
        for _ in ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
            model_dict[_] = supervised
        # ResNet and FTTransformer for tabular data
        for _ in ['ResNet', 'FTTransformer']:
            model_dict[_] = FTTransformer

    else:
        raise NotImplementedError

    # We remove the following model for considering the computational cost
    for _ in ['SOGAAL', 'MOGAAL', 'LSCP', 'MCD', 'FeatureBagging']:
        if _ in model_dict.keys():
            model_dict.pop(_)
    return model_dict


class ADBenchModelWrapper():
    def __init__(self, model_name: str, learning_mode: str, tuning: bool = False, ):
        model_dict = generate_AD_dictionnary(learning_mode) 
        self.model = self.model_dict[self.model_name]