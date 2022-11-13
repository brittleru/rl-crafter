import os


def get_dir_name_as_img(path: str, img_type: str = "png") -> str:
    return f"{os.path.basename(os.path.normpath(path))}.{img_type}"
