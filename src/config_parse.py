
from pathlib import Path

import Hyperparameters
from DetectionAlgorithm import DetectionAlgorithm, ModelType
from utils.toml_utils import load_toml
from utils.write_data import save_path


def read_model_config(config_file):
    """ Parse config file for models."""
    config_table = load_toml(config_file)
    default_save_path = save_path(config_table['save-root'])
    models = config_table['models']
    algs = list()
    for model in models:
        hp = model['hyperparameters']
        model_type: str = model['type']
        model_name: str = model['name']
        if 'save-path' in model:
            save_dir = save_path(model['save-path'])
        else:
            save_dir = default_save_path
        m_save_path = Path(save_dir, model['save-name'])
        if 'show-progress' in model:
            with_progress = model['show-progress']
        else:
            with_progress = False
        match model_type:
            case 'bocpd':
                hyperparams = Hyperparameters.BOCPDHyperparams(
                        alpha=hp['alpha'], beta=hp['beta'], mu=hp['mu'],
                        kappa=hp['kappa'], lamb=hp['lambda'])
            case 'expectation maximization':
                hyperparams = Hyperparameters.EMHyperparams(
                        normal_data_size=hp['normal-data-size'],
                        abnormal_data_size=hp['abnormal-data-size'],
                        normal_mean=hp['normal-mean'],
                        abnormal_mean=hp['abnormal-mean'],
                        normal_var=hp['normal-variance'],
                        abnormal_var=hp['abnormal-variance'],
                        pi=hp['pi'], epochs=hp['epochs'])
            case 'cusum':
                hyperparams = Hyperparameters.CUSUMHyperparams(
                        mean=hp['mean'], std_dev=hp['standard-deviation'], h=hp['h'],
                        alpha=hp['alpha'])
            case 'grey':
                hyperparams = Hyperparameters.GreyHyperparams(
                        window_size=hp['window-size'],
                        critical_value=hp['critical-value'],
                        critical_ratio_value=hp['critical-ratio-value'],
                        alpha=hp['alpha'])
            case 'nonparametric':
                hyperparams = Hyperparameters.NonparametricHyperparams(
                        window_size=hp['window-size'],
                        critical_value=hp['critical-value'], alpha=hp['alpha']
                    )
            case _:
                raise NotImplementedError
        alg = DetectionAlgorithm(
            type=ModelType(model_type),
            name=model_name, with_progress=with_progress,
            save_path=m_save_path, hyperparameters=hyperparams)
        algs.append(alg)
    return algs