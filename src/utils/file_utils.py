import os
from pathlib import PurePath


def create_dir_if_doesnt_exist(path_to_dir: str, is_silent: bool = True) -> bool:
    """
    This will create the directories if they are not already existing.

    :param path_to_dir: The full path for the new directory.
    :param is_silent: If set to true it will display info about the directory.
    :return: A binary value that represents if the directories were created or not.
    """
    dir_exists = os.path.exists(path_to_dir)

    if not dir_exists:
        os.makedirs(path_to_dir, exist_ok=False)
        if not is_silent:
            print(f"Successfully created '{PurePath(path_to_dir).name}' directory.")
        return True

    if not is_silent:
        print(f"Directory '{PurePath(path_to_dir).name}' already exists.")
    return False
