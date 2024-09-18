import os
from pathlib import Path


def save_path(path_name):
    if isinstance(path_name, Path):
        return path_name
    elif path_name is None or path_name == '':
        return Path(os.curdir)
    elif isinstance(path_name, str) and os.path.exists(path_name):
        return Path(path_name)
    raise ValueError
