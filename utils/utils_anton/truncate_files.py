"""
@author: Hailin Shi
@date: 202212
@contact: hailinshi.work@outlook.com

note: TBD
"""

from utils_anton import *


def truncating(original_path, truncated_path, trunct_length):

    assert os.path.exists(original_path)
    assert not os.path.exists(truncated_path), "truncated file list already exists!"
    trunct_length = int(trunct_length)
    pre_path, base_name = os.path.split(truncated_path)
    if not os.path.exists(pre_path):
        os.makedirs(pre_path)

    label_set = set()
    original_file_buf = open(original_path)
    f = open(truncated_path, 'a')

    while len(label_set) <= trunct_length:
        line = original_file_buf.readline().strip()
        image_path, label = line.split()
        label_set.add(label)
        f.write(line + '\n')
        print(len(label_set))
    f.close()
    original_file_buf.close()

    f = open(truncated_path, 'rb+')
    lines = f.readlines()
    f.seek(-len(lines[-1]), os.SEEK_END)
    f.truncate()
    f.close()


if __name__ == "__main__":
    original_path, truncated_path, trunct_length = \
        sys.argv[1], sys.argv[2], sys.argv[3]

    truncating(original_path, truncated_path, trunct_length)

    pass


