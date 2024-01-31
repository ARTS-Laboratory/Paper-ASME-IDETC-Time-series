""" Thompson 2022"""
import tomlkit


def load_toml(filename: str):
    """

    :param filename: Path to TOML file.
    :return: TOML Document located at filepath
    """
    with open(filename, 'r') as toml_file:
        data = tomlkit.load(toml_file)
    return data
