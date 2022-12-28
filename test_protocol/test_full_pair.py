"""
@author: Hailin Shi
@date: 202212
@contact: hailinshi.work@outlook.com
Inherited from test_lfw.py by Jun Wang
"""

import argparse
import os.path
import pickle

import yaml
from torch.utils.data import DataLoader
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
from anton.full_pair_evaluator import FullPairEvaluator
from prettytable import PrettyTable

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='full pair test protocol.')
    conf.add_argument("--test_set", type=str,
                      help="lfw.")
    conf.add_argument("--data_conf_file", type=str,
                      help="the path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type=str,
                      help="Resnet, Mobilefacenets..")
    conf.add_argument("--backbone_conf_file", type=str,
                      help="The path of backbone_conf.yaml.")
    conf.add_argument('--batch_size', type=int, default=1024)
    conf.add_argument('--model_path', type=str, default='mv_epoch_8.pt',
                      help='The path of model or the directory which some models in.')
    args = conf.parse_args()
    # parsing configuration
    far = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f, Loader=yaml.FullLoader)[args.test_set]
        cropped_face_folder = data_conf['cropped_face_folder']
        image_list_file_path = data_conf['image_list_file_path']
    test_set = args.test_set
    model_path = args.model_path
    # define dataloader
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False),
                             batch_size=args.batch_size, num_workers=1, shuffle=False)
    # define model and extractor
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    model_loader = ModelLoader(backbone_factory)
    feature_extractor = CommonExtractor('cuda:0')
    # load model(s) and run test
    if not os.path.isdir(model_path):
        res_save_path = os.path.split(model_path)[0]
        # define evaluator
        fp_evaluator = FullPairEvaluator(data_loader, feature_extractor,
                                         image_list_file_path, test_set)
        model = model_loader.load_model(model_path)
        vrs, thresholds = fp_evaluator.test(model)
        vrs_list = [(os.path.basename(model_path), vrs, thresholds), ]
        model_name_list = [os.path.basename(model_path), ]
    else:
        res_save_path = model_path
        vrs_list = []
        model_name_list = os.listdir(model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                print(model_name)
                # define evaluator
                fp_evaluator = FullPairEvaluator(data_loader, feature_extractor,
                                                 image_list_file_path, test_set)
                model_path_this = os.path.join(model_path, model_name)
                model = model_loader.load_model(model_path_this)
                vrs, thresholds = fp_evaluator.test(model, far)
                vrs_list.append((model_name, vrs, thresholds))
    pretty_tabel = PrettyTable(["model_name", "vrs", "thresholds"])
    for accu_item in vrs_list:
        pretty_tabel.add_row(accu_item)
    print(pretty_tabel)

    res = (model_name_list, vrs_list, far)
    f = open(os.path.join(res_save_path, 'res_vrs_far_thr.pkl'), 'wb')
    pickle.dump(res, f)
    f.close()











