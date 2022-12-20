"""
@author: Hailin Shi, Jia Shen
@date: 202212
@contact: hailinshi.work@outlook.com

note:
To make lmdb from raw data with specific image file list.
The image is stored in JPEG format.
The structure in LMDB:
LMDB
\_key: label int
\_value: a list containing the images in this subject, e.g. [img1, img2, ...]

diction
\_key: label int
\_value: a list containing the image paths in this subject, e.g. [path1, path1, ...]

"""



from utils_anton import *
from io import BytesIO


def make_lmdb(raw_img_root, file_list_path, save_path, save_name):
    contn = check_label_continue(file_list_path)
    if not contn:
        file_list_path = squeeze_file_label(file_list_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    lmdb_path = os.path.join(save_path, save_name)
    dict_path = os.path.join(save_path, save_name + "_dict")
    env = lmdb.open(lmdb_path, map_size=MAX_LMDB_SIZE)
    dict_file = {}

    file_buf = open(file_list_path)
    line = file_buf.readline().strip()
    label_curr, label_prev = 0, -1
    images = []
    paths = []
    while line:
        image_path, label = line.split()
        label_curr = int(label)
        image_path_total = os.path.join(raw_img_root, image_path)
        img_open = Image.open(image_path_total)
        img_open = img_open.convert("RGB")
        bytesio = BytesIO()
        img_open.save(bytesio, format="JPEG")

        if not label_curr == label_prev:
            assert label_curr == label_prev + 1, "the file list does not satisfy the requirement!"
            print("make lmdb with label %d" % label_curr)
            if label_curr == 0:
                pass
                images = []
                paths = []
                images.append(bytesio.getvalue())
                paths.append(image_path)
                label_prev = label_curr
            else:
                pass
                # write this subject into lmdb
                k = str(label_prev)
                v = pickle.dumps(images)
                txn = env.begin(write=True)
                txn.put(k.encode(), v)
                txn.commit()
                dict_file[label_prev] = paths
                # clear the list, store the next subject
                images = []
                paths = []
                images.append(bytesio.getvalue())
                paths.append(image_path)
                label_prev = label_curr
        else:
            images.append(bytesio.getvalue())
            paths.append(image_path)

        line = file_buf.readline().strip()

    env.close()
    file_buf.close()

    with open(dict_path, 'wb') as f:
        pickle.dump(dict_file, f)

    print("LMDB done.")



if __name__ == "__main__":
    raw_img_root, file_list_path, save_path, save_name = \
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    make_lmdb(raw_img_root, file_list_path, save_path, save_name)
