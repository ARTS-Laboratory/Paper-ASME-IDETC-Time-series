import os

from pathlib import Path

from enum import Enum

class CreatDirOptions(Enum):
    ACCEPT = 0
    DENY = 1
    ABORT = 2

def check_dir_exists(path: Path):
    """ """
    path_exists = path.exists()
    path_is_dir = path.is_dir()
    return path_exists and path_is_dir

# def validate_dir(dir: Path, allow_make=False)

def confirm_dir_or_consult(dir: Path):
    """ Confirm directory is correct or consult with user."""
    if check_dir_exists(dir):
        return True
    question = (f'The directory provided cannot be found.'
                f' Would you like to create the directory?\nDirectory to be '
                f'created: {dir.resolve(strict=False)}')
    user_accepts = ask_user_default_no(question)
    match user_accepts:
        case CreatDirOptions.ACCEPT:
            dir.mkdir(parents=True, exist_ok=False)
            return True
        case CreatDirOptions.DENY:
            return False
        case CreatDirOptions.ABORT:
            raise FileNotFoundError(f'The directory ("{dir.resolve(strict=False)}") could not be found.')

def ask_user_default_no(question: str) -> CreatDirOptions:
    """ Ask the user a question, and return answer."""
    options = '(Y/n/abort (default)):\n'
    q_with_options = ' '.join((question, options))
    answer = input(q_with_options)
    match answer:
        case 'Y':
            decision = CreatDirOptions.ACCEPT
        case 'n':
            decision = CreatDirOptions.DENY
        case '' | 'abort':
            # User chose to abort
            decision = CreatDirOptions.ABORT
        case _:
            # user gave bad input, default to abort
            decision = CreatDirOptions.ABORT
    return decision
