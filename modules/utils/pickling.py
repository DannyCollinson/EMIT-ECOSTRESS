import os
import pickle
from typing import Any

def join_path(project_path: str, relative_path: str) -> str:
    '''Utility for making working with paths within the project easier'''
    return os.path.join(project_path, relative_path)

def pickle_load(
        project_path: str, relative_path: str
) -> Any:  # -> pickled_file_contents
    '''Returns the contents of the pickled file at the specified path'''
    return pickle.load(open(join_path(project_path, relative_path), 'rb'))

def pickle_save(project_path: str, obj: object, relative_path: str) -> None:
    '''Saves the provided object to a pickle file at the provided path'''
    pickle.dump(obj, open(join_path(project_path, relative_path), 'wb'))