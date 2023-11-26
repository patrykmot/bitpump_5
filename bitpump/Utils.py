import os


def create_folder(folder_path: str) -> bool:
    if not is_file_exist(folder_path):
        os.makedirs(folder_path)
        return True
    return False


def is_file_exist(file_path: str) -> bool:
    return os.path.exists(file_path)


def join_file_system_path(p1: str, p2: str):
    return os.path.join(p1, p2)
