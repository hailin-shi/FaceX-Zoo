"""
@author: Hailin Shi
@date: 202212
@contact: hailinshi.work@outlook.com

note: TBD
"""




import os
import sys
import time
import cv2
import time
import pickle
from PIL import Image
import lmdb
import numpy as np

MAX_LMDB_SIZE = 1024**4
# MAX_LMDB_SIZE = 42949672960

def check_label_continue(file_path, label_col=2):
    assert os.path.exists(file_path), "the file does not exist"
    label_col = int(label_col)
    file_buf = open(file_path)
    line = file_buf.readline().strip()
    label_curr = int(line.split()[label_col-1])
    print("the initial label is %d" % label_curr)
    time.sleep(5)

    contn = True

    while line:
        line = file_buf.readline().strip()
        label_prev = label_curr
        if line:
            label_curr = int(line.split()[label_col - 1])
        print(label_curr)
        if (label_prev == label_curr) or (label_prev == label_curr-1):
            continue
        else:
            print("the labels in this file breaks in %d" % label_prev)
            file_buf.close()
            contn = False
            break

    if contn:
        print("done, this file has continual labels")
        file_buf.close()
    else:
        print("done")

    return contn


def squeeze_file_label(file_path):
    assert os.path.exists(file_path), "the file does not exist"
    pre_path, base_name = os.path.split(file_path)
    file_path_sqeeze = os.path.join(pre_path, base_name.split(".")[0] + "_squeezed" + base_name.split(".")[1])

    label_set = set()
    real_label = -1
    real_line = []
    label_dict = {}
    file_buf = open(file_path)
    f = open(file_path_sqeeze, 'a')

    line = file_buf.readline().strip()
    while line:
        image_path, label = line.split()
        if not label in label_set:
            label_set.add(label)
            real_label = real_label + 1
            label_dict[label] = real_label
            print(real_label)
        else:
            real_label = label_dict[label]
        real_line = image_path + " " + str(real_label)
        f.write(real_line + '\n')
        line = file_buf.readline().strip()
    f.close()
    return file_path_sqeeze

if __name__ == "__main__":
    file_path = sys.argv[1]
    contn = check_label_continue(file_path)
    if not contn:
        squeeze_file_label(file_path)







