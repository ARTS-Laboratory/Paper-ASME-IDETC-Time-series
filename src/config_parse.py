
from pathlib import Path

import Hyperparameters
from DetectionAlgorithm import DetectionAlgorithm, ModelType
from utils.path_validation import confirm_dir_or_consult
from utils.toml_utils import load_toml
from utils.write_data import save_path


def read_model_config(config_file):
    """ Parse config file for models."""
    config_table = load_toml(config_file)
    default_save_path = Path(config_table['save-root'])
    confirm_dir_or_consult(default_save_path)
    models = config_table['models']
    algs = list()
    for model in models:
        hp = model['hyperparameters']
        model_type: str = model['type']
        model_name: str = model['name']
        if 'save-path' in model:
            save_dir = Path(model['save-path'])
            confirm_dir_or_consult(save_dir)
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
                    alpha=hp['alpha'].unwrap(), beta=hp['beta'].unwrap(),
                    mu=hp['mu'].unwrap(), kappa=hp['kappa'].unwrap(),
                    lamb=hp['lambda'].unwrap())
            case 'expectation maximization':
                hyperparams = Hyperparameters.EMHyperparams(
                        normal_data_size=hp['normal-data-size'].unwrap(),
                        abnormal_data_size=hp['abnormal-data-size'].unwrap(),
                        normal_mean=hp['normal-mean'].unwrap(),
                        abnormal_mean=hp['abnormal-mean'].unwrap(),
                        normal_var=hp['normal-variance'].unwrap(),
                        abnormal_var=hp['abnormal-variance'].unwrap(),
                        pi=hp['pi'].unwrap(), epochs=hp['epochs'].unwrap())
            case 'cusum':
                hyperparams = Hyperparameters.CUSUMHyperparams(
                    mean=hp['mean'].unwrap(),
                    std_dev=hp['standard-deviation'].unwrap(),
                    h=hp['h'].unwrap(),
                    alpha=hp['alpha'].unwrap())
            case 'grey':
                hyperparams = Hyperparameters.GreyHyperparams(
                    window_size=hp['window-size'].unwrap(),
                    critical_value=hp['critical-value'].unwrap(),
                    critical_ratio_value=hp['critical-ratio-value'].unwrap(),
                    alpha=hp['alpha'].unwrap())
            case 'nonparametric':
                hyperparams = Hyperparameters.NonparametricHyperparams(
                    window_size=hp['window-size'].unwrap(),
                    critical_value=hp['critical-value'].unwrap(),
                    alpha=hp['alpha'].unwrap()
                    )
            case _:
                raise NotImplementedError
        alg = DetectionAlgorithm(
            type=ModelType(model_type),
            name=model_name, with_progress=with_progress,
            save_path=m_save_path, hyperparameters=hyperparams)
        algs.append(alg)
    return algs


# def parse_models_config(models_config: dict):
#     """ Parse models config and return model config object."""
#     algs = list()
#     for model in models:
#         hp = model['hyperparameters']
#         model_type: str = model['type']
#         model_name: str = model['name']
#         save_name: str = model['save-name']
#         match model_type:
#             case 'bocpd':
#                 hyperparams = Hyperparameters.BOCPDHyperparams(
#                         alpha=hp['alpha'], beta=hp['beta'], mu=hp['mu'],
#                         kappa=hp['kappa'], lamb=hp['lambda'])
#             case 'expectation maximization':
#                 hyperparams = Hyperparameters.EMHyperparams(
#                         normal_data_size=hp['normal-data-size'],
#                         abnormal_data_size=hp['abnormal-data-size'],
#                         normal_mean=hp['normal-mean'],
#                         abnormal_mean=hp['abnormal-mean'],
#                         normal_var=hp['normal-variance'],
#                         abnormal_var=hp['abnormal-variance'],
#                         pi=hp['pi'], epochs=hp['epochs'])
#             case 'cusum':
#                 hyperparams = Hyperparameters.CUSUMHyperparams(
#                         mean=hp['mean'], std_dev=hp['standard-deviation'], h=hp['h'],
#                         alpha=hp['alpha'])
#             case 'grey':
#                 hyperparams = Hyperparameters.GreyHyperparams(
#                         window_size=hp['window-size'],
#                         critical_value=hp['critical-value'],
#                         critical_ratio_value=hp['critical-ratio-value'],
#                         alpha=hp['alpha'])
#             case 'nonparametric':
#                 hyperparams = Hyperparameters.NonparametricHyperparams(
#                         window_size=hp['window-size'],
#                         critical_value=hp['critical-value'], alpha=hp['alpha']
#                     )
#             case _:
#                 # should i crash the whole app here or skip the model?
#                 raise NotImplementedError
#         alg = DetectionAlgorithm(
#             type=ModelType(model_type),
#             name=model_name, with_progress=with_progress,
#             save_path=m_save_path, hyperparameters=hyperparams)
#         algs.append(alg)
#     return algs


