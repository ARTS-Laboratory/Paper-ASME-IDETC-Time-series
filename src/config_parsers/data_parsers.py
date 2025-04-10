
from enum import Enum
from pathlib import Path

from utils.read_data import load_signals


class DataSource(Enum):
    ARRAY = 'array'

def parse_data_config(config: dict):
    """ Parse data config and return Data object."""
    what = config['what']  # what type of data to expect
    where = config['where']  # the table of information we need to retrieve the data
    data_source = DataSource(what)
    match what:
        case DataSource.ARRAY:
            data_dir = where['dir']
            data_name = where['filename']
            file_path = Path(data_dir, data_name)
            load_signals(file_path)
        case _:
            raise NotImplementedError(f"DataSource {what} not supported")
