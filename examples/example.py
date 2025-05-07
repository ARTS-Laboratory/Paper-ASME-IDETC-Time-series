import os
from pathlib import Path

from config_parse import read_model_config
from main import make_ground_truth, process_online_models
from utils.detection_arr_helpers import convert_interval_indices_to_full_arr
from utils.matplotlib_formatting import set_rc_params
from utils.read_data import load_signals
from utils.toml_utils import load_toml

EXAMPLE_CONFIG = Path(os.pardir, 'configs', 'example-main-config.toml')

def run_from_example_config(config_file_name):
    """ """
    set_rc_params()
    config_table = load_toml(config_file_name)
    file_path = Path(config_table['file-path'])
    time, data = load_signals(file_path)
    algs = read_model_config(config_file_name)
    ## get ground truth
    (true_shocks, true_nonshocks) = make_ground_truth(time, data)
    ground = convert_interval_indices_to_full_arr(true_shocks, true_nonshocks, len(time))
    ## Process online models
    df = process_online_models(time, data, algs, ground)
    print(df)

def main():
    run_from_example_config(EXAMPLE_CONFIG)

if __name__ == '__main__':
    main()