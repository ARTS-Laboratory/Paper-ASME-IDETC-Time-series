import os
from pathlib import Path


def save_path(path_name):
    """ Convert path_name to Path object.

        :param path_name: Path to convert.
        :type path_name: str | Path | None
        :return: Converted path.
    """
    if isinstance(path_name, Path):
        return path_name
    elif path_name is None or path_name == '':
        return Path(os.curdir)
    elif isinstance(path_name, str) and os.path.exists(path_name):
        return Path(path_name)
    raise ValueError(f'{path_name} is not a valid or existing path.')
