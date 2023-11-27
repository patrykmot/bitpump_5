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


def add_postfix(list_in: [], postfix: str) -> []:
    list_postfix: [] = []
    for s in list_in:
        list_postfix.append(s + postfix)
    return list_postfix
