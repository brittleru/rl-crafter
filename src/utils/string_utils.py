import os
from typing import List


def get_dir_name_as_img(path: str, img_type: str = "png") -> str:
    return f"{os.path.basename(os.path.normpath(path))}.{img_type}"


def get_dirs_name_as_img(paths: List[str], img_type: str = "png") -> str:
    file_name = ""
    for path in paths:
        file_name += f"{os.path.basename(os.path.normpath(path))}_"
    return f"{file_name[:-1]}_all_results.{img_type}"
