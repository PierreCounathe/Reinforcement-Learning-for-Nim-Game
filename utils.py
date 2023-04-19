import os

def check_create_folder(relative_path_to_folder):
    if not os.path.exists(relative_path_to_folder):
        os.mkdir(relative_path_to_folder)