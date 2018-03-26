import os


def get_file(root_dir):
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        print(dir_name.split(os.sep))
        c = 0
        for f in file_list:
            yield dir_name + os.sep + f, \
                  dir_name.split(os.sep)[-1], \
                  dir_name.split(os.sep)[-2]
            c += 1


def get_checkpoints_list(root_dir):
    checkpoints = list()
    for file in sorted(os.listdir(root_dir)):
        if file.startswith('model_') and file.endswith('.ckpt.meta'):
            checkpoints.append(file[6:-10])
    return checkpoints


def separate_by_os(path):
    path = path.replace('/', os.sep)
    return path.replace('\\', os.sep)


def shrink_file_name(file_name, file_name_len):
    if len(file_name) > file_name_len:
        file_name = file_name[0:(2 * file_name_len // 3)] + '...' + file_name[(-file_name_len // 3):]
    return file_name


def end_with_sep(path: str):
    if path.endswith(os.sep):
        return path
    return path + os.sep
