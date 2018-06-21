import os
import argparse


def del_dir_tree(root):
    """
    remove dir trees
    :param path: root dir
    :return: True or False
    """
    if os.path.isfile(root):
        try:
            os.remove(root)
        except Exception as e:
            print(e)
    elif os.path.isdir(root):
        for item in os.listdir(root):
            itempath = os.path.join(root, item)
            del_dir_tree(itempath)
        try:
            os.rmdir(root)
            print('Files in {} is removed'.format(root))
        except Exception as e:
            print(e)


def make_dir_tree(path):
    """
    make dir trees
    :param path: abstract dir path
    :return:
    """
    isExists = os.path.exists(path)
    if not isExists:
        try:
            os.makedirs(path)
            print('Making dirs {} succesfully'.format(path))
        except os.error:
            print('Making dirs {} failed'.format(path))
        return True
    else:
        print('{} is already existed'.format(path))
        return False


def get_dir_tree(root):
    """
    Abstract path for all files
    :param root:root dir
    :return:List of all abstract files path
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    return files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--option', type=str, help='option', default=None)
    parser.add_argument('--path', type=str, help='The dir path', default=None)
    args = parser.parse_args()
    if args.option=='del':
        print('delet dir tree '+args.path)
        del_dir_tree(args.path)
    if args.option=='get':
        print('get files '+args.path)
        get_dir_tree(args.path)
    if args.option=='make':
        print('get dir tree '+args.path)
        make_dir_tree(args.path)