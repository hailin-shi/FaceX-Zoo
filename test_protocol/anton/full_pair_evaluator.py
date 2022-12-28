"""
@author: Hailin Shi, Jianzhu Guo
@date: 202212
@contact: hailinshi.work@outlook.com
"""

import re
import math
import numpy as np
from collections import OrderedDict, Iterable
from math import sqrt

TEST_BASES = ['LFW', ]
PTRN_SPSH = re.compile('/')

class FullPairEvaluator(object):
    """
        Evaluation with ROC protocol with full matched pairs
    """
    def __init__(self, data_loader, feature_extractor, file_list, test_set='LFW'):
        self.data_loader = data_loader
        self.feature_extractor = feature_extractor
        self.test_set = test_set
        self.file_list = file_list
        self.feat_dim, self.num_gallery, self.num_probe = None, None, None
        self.id_dict = OrderedDict()
        assert self.test_set in TEST_BASES

    def parse_file_list(self, file_list):
        if self.test_set == 'LFW':
            self._lfw_parse_file_list(file_list)
        else:
            raise Exception("no such test set")
        return True

    def _lfw_parse_file_list(self, file_list):
        file_list_buf = open(file_list)
        line = file_list_buf.readline().strip()
        count_file = 0

        while line:
            id_name, file_name = PTRN_SPSH.split(line)
            if not id_name in self.id_dict.keys():
                self.id_dict[id_name] = [file_name, ]
            else:
                self.id_dict[id_name].append(file_name)
            count_file += 1
            line = file_list_buf.readline().strip()
        self.num_gallery = len(self.id_dict)
        self.num_probe = count_file - self.num_gallery
        return True

    def find_kth_largest_element(self, score_arr, k):
        """return element value and the real far"""
        k = int(math.floor(k))
        assert 0 <= k <= score_arr.size - 1
        if k == 0:
            return np.max(score_arr), np.max(score_arr), 0., 0.

        threshold = 0.4
        while score_arr[score_arr > threshold].size < k:
            threshold -= 0.05
        arr_sort = np.sort(score_arr[score_arr > threshold])
        # return arr_sort[-(k + 1)], k / arr.size
        return arr_sort[-(k + 2)], arr_sort[-(k + 1)], (k + 1) / score_arr.size, k / score_arr.size

    def compute_roc(self, score, label, far=(1e-4, 1e-5, 1e-6, 1e-7, 1e-8)):
        score_pos = score[label == 1]
        score_neg = score[label == 0]

        if not isinstance(far, Iterable):
            far = [far]

        vrs = []
        threshold_aves = []
        for fr in far:
            kth_score1, kth_score2, far_real1, far_real2 = self.find_kth_largest_element(score_neg, fr * score_neg.size)
            pos_num = len(score_pos)
            pos_valid_num1 = len(score_pos[score_pos > kth_score1])
            pos_valid_num2 = len(score_pos[score_pos > kth_score2])

            vr_ave = (pos_valid_num1 + pos_valid_num2) / 2 / pos_num
            threshold_ave = (kth_score1 + kth_score2) / 2
            vrs.append(vr_ave)
            threshold_aves.append(threshold_ave)
        return vrs, threshold_aves



    def test(self, model, far=(1e-4, 1e-5, 1e-6, 1e-7, 1e-8)):
        # 1. extract features and get dimension
        image_name2feature = self.feature_extractor.extract_online(model, self.data_loader)
        self.feat_dim = image_name2feature[list(image_name2feature.keys())[0]].size
        assert self.feat_dim > 0

        # 2. parse and initialize the matrices of gallery and probe
        # get ready self.feat_dim, self.num_gallery, self.num_probe, self.id_dict
        self.parse_file_list(self.file_list)
        mtx_gallery = np.zeros((self.num_gallery, self.feat_dim))
        mtx_probe = np.zeros((self.num_probe, self.feat_dim))

        # 3. build label matrix
        # define the first image in each list as the gallery of this id, and the
        # remaining ones go to the probe
        mtx_label = np.zeros((self.num_gallery, self.num_probe), dtype=np.int8)
        raw_probe_label = np.zeros(self.num_probe, dtype=np.uint32)
        raw_gallery_label = np.zeros(self.num_probe, dtype=np.uint32)
        id_names = list(self.id_dict.keys())
        cursor, curr_label = 0, 0
        for id_name in id_names:
            file_names = self.id_dict[id_name]
            num_probe_curr_id = len(file_names) - 1
            raw_probe_label[cursor : cursor + num_probe_curr_id] = curr_label
            cursor = cursor + num_probe_curr_id
            curr_label += 1
        curr_label = 0
        for i in range(self.num_gallery):
            raw_gallery_label.fill(curr_label)
            curr_label += 1
            mtx_label[i] = raw_gallery_label == raw_probe_label

        # 4 compute scores
        l2_norm = lambda feat: feat / sqrt(feat.dot(feat))
        cursor_g, cursor_p = 0, 0
        for id_name in id_names:
            file_names = self.id_dict[id_name]
            # aggregate gallery
            file_name = file_names[0]
            feat_key = id_name + '/' + file_name
            feat = l2_norm(image_name2feature[feat_key])
            mtx_gallery[cursor_g] = feat
            cursor_g += 1
            # aggregate probe
            if len(file_names) > 1:
                for file_name in file_names[1:]:
                    feat_key = id_name + '/' + file_name
                    feat = l2_norm(image_name2feature[feat_key])
                    mtx_probe[cursor_p] = feat
                    cursor_p += 1
        assert cursor_g == self.num_gallery
        assert cursor_p == self.num_probe
        score = mtx_gallery.dot(mtx_probe.transpose()).astype(np.float32)

        # 5. compute ROC
        vrs, thresholds = self.compute_roc(score, mtx_label, far)
        return vrs, thresholds









