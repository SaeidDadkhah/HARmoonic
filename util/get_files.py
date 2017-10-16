import os


def get_file():
    root_dir = './data/'
    for dir_name, subdir_list, file_list in os.walk(root_dir):
        print(dir_name.split(os.sep))
        c = 0
        for f in file_list:
            yield dir_name + os.sep + f,\
                  dir_name.split(os.sep)[3],\
                  dir_name.split(os.sep)[2]
            c += 1
            if c > 5:
                break
