"""ADBench model registry — maps learning modes to detector classes."""


def generate_AD_dictionary(learning_mode: str) -> dict:
    model_dict = {}

    if learning_mode == "unsupervise":
        from adbench.baseline.PyOD import PYOD
        from adbench.baseline.DAGMM.run import DAGMM
        for name in ['IForest', 'OCSVM', 'CBLOF', 'COF', 'COPOD', 'ECOD',
                      'FeatureBagging', 'HBOS', 'KNN', 'LODA', 'LOF', 'LSCP',
                      'MCD', 'PCA', 'SOD', 'SOGAAL', 'MOGAAL', 'DeepSVDD']:
            model_dict[name] = PYOD
        model_dict['DAGMM'] = DAGMM

    elif learning_mode == "semi-supervise":
        from adbench.baseline.PyOD import PYOD
        from adbench.baseline.GANomaly.run import GANomaly
        from adbench.baseline.DeepSAD.src.run import DeepSAD
        from adbench.baseline.REPEN.run import REPEN
        from adbench.baseline.DevNet.run import DevNet
        from adbench.baseline.PReNet.run import PReNet
        from adbench.baseline.FEAWAD.run import FEAWAD
        model_dict = {
            'GANomaly': GANomaly, 'DeepSAD': DeepSAD, 'REPEN': REPEN,
            'DevNet': DevNet, 'PReNet': PReNet, 'FEAWAD': FEAWAD, 'XGBOD': PYOD,
        }

    elif learning_mode == "supervise":
        from adbench.baseline.Supervised import supervised
        from adbench.baseline.FTTransformer.run import FTTransformer
        for name in ['LR', 'NB', 'SVM', 'MLP', 'RF', 'LGB', 'XGB', 'CatB']:
            model_dict[name] = supervised
        for name in ['ResNet', 'FTTransformer']:
            model_dict[name] = FTTransformer

    else:
        raise NotImplementedError(f"Unknown learning_mode: {learning_mode}")

    for name in ['SOGAAL', 'MOGAAL', 'LSCP', 'MCD', 'FeatureBagging']:
        model_dict.pop(name, None)

    return model_dict


class ADBenchModelWrapper:
    def __init__(self, model_name: str, learning_mode: str, tuning: bool = False):
        model_dict = generate_AD_dictionary(learning_mode)
        if model_name not in model_dict:
            raise KeyError(f"Model '{model_name}' not in {learning_mode} registry")
        self.model = model_dict[model_name]
