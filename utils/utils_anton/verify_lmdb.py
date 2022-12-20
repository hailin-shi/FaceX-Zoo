"""
@author: Hailin Shi
@date: 202212
@contact: hailinshi.work@outlook.com

note: TBD
"""
import pickle

import cv2

from utils_anton import *
from io import BytesIO

def verify_lmdb(raw_img_root, file_list_path, save_path, save_name):

    ## the index of the sample to be checked
    id = 7

    ## read the first image via raw image file
    file_buf = open(file_list_path)
    for i in range(id):
        line = file_buf.readline().strip()
    file_buf.close()
    image_path, label = line.split()
    image_path_total = os.path.join(raw_img_root, image_path)
    image1 = cv2.imread(image_path_total)
    print("image1 path: %s" % image_path)



    ## read the first image via lmdb & diction
    dict_path = os.path.join(save_path, save_name + "_dict")
    with open(dict_path, 'rb') as f:
        dict_file = pickle.load(f)
    image2_path = dict_file[0][id-1]
    print("image2 path: %s" % image2_path)
    lmdb_path = os.path.join(save_path, save_name)
    env = lmdb.open(lmdb_path)
    txn = env.begin()
    images = pickle.loads(txn.get(str(0).encode()))
    image_byte = images[id-1]
    image_nparr = np.frombuffer(image_byte, np.uint8)
    image2 = cv2.imdecode(image_nparr, cv2.IMREAD_COLOR)

    count = 0
    for i in range(image1.shape[0]):
        for j in range(image1.shape[1]):
            for k in range(image1.shape[2]):
                i_1 = image1[i, j, k]
                i_2 = image2[i, j, k]

                if not i_1 == i_2:
                    count = count + 1
                # else:
                #     print("i: {}, j: {}, k: {}".format(i, j, k))
                    print("i_1: {}, i_2: {}".format(i_1, i_2))

    print(count)
    print(count/(image1.size))

    pass
    env.close()



if __name__ == "__main__":
    raw_img_root, file_list_path, save_path, save_name = \
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    verify_lmdb(raw_img_root, file_list_path, save_path, save_name)