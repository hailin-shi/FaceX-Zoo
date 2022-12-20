from parse_log_hailin import *
import os
import glob
from matplotlib import pyplot as plt
import sys

def show_curves_multi_file(log_files, loss_names):

    if not isinstance(log_files, list):
        raise Exception('input should be either list or tuple.')
    color = ['b', 'g', 'c', 'm', 'y', 'k'] * 10
    if len(log_files) > len(color)-1:
        raise Exception('#max supported log file is %d.' % len(color))

    num_loss = len(loss_names)
    unit_size = 8
    figure = plt.figure(figsize=[unit_size, unit_size])
    figure.add_subplot(1, 1, 1)
    min_y_global, max_y_global = 99999999, -99999999

    list_dict_train, list_dict_test, list_iters, list_key = \
        [], [], [], []
    for log_filename in log_files:
        _, filename = os.path.split(log_filename)
        key, _ = os.path.splitext(filename)

        dict_train, dict_test = init_dicts(loss_names)
        dict_train, dict_test, iters = parse_log(log_filename, dict_train, dict_test, loss_names)

        list_dict_train.append(dict_train)
        list_dict_test.append(dict_test)
        list_iters.append(iters)
        list_key.append(key)

    for loss_id in range(num_loss):
        loss_name = loss_names[loss_id]
        curves_dict = {}

        min_y = 1e20
        max_y = 1e-20

        for file_id in range(len(log_files)):
            dict_train, dict_test, iters, key = \
                list_dict_train[file_id], list_dict_test[file_id], list_iters[file_id], list_key[file_id]
            assert not key in curves_dict
            curves_dict[key] = {}

            curves_dict[key]['x_train'] = []
            curves_dict[key]['y_train'] = []
            curves_dict[key]['x_test'] = []
            curves_dict[key]['y_test'] = []

            for iter in iters:
                if iter in dict_train[loss_name].keys():
                    loss_value = dict_train[loss_name][iter]
                    curves_dict[key]['x_train'].append(iter)
                    curves_dict[key]['y_train'].append(loss_value)

                    if min_y > loss_value:
                        min_y = loss_value
                    if max_y < loss_value:
                        max_y = loss_value

                if iter in dict_test[loss_name].keys():
                    loss_value = dict_test[loss_name][iter]
                    curves_dict[key]['x_test'].append(iter)
                    curves_dict[key]['y_test'].append(loss_value)

                    if min_y > loss_value:
                        min_y = loss_value
                    if max_y < loss_value:
                        max_y = loss_value

                else:
                    pass
                    # assert False

        if min_y < min_y_global:
            min_y_global = min_y
        if max_y > max_y_global:
            max_y_global = max_y

        plt.ylim([min_y_global, max_y_global])
        plt.xlabel('iterations')
        plt.ylabel(loss_name)
        for k, v in curves_dict.iteritems():
            if 'x_train' in v:
                # plt.plot(v['x_train'], v['y_train'], color[i], linewidth=2, label=k + '_train')
                plt.plot(v['x_train'], v['y_train'], color[loss_id], linewidth=2)
            if 'x_test' in v:
                # plt.plot(v['x_test'], v['y_test'], color[-1], linewidth=2, label=k + '_test')
                plt.plot(v['x_test'], v['y_test'], color[loss_id], linewidth=2, linestyle='--')

    loss_name_full = []
    for loss_name in loss_names:
        loss_name_full.append(loss_name + '-train')
        loss_name_full.append(loss_name + '-test')

    plt.title(''.join(['multiple indicators', ' vs ', 'iterations']))
    plt.legend(loss_name_full, loc='best')
    plt.grid()

    plt.show()


if __name__ == '__main__':

    subdir = sys.argv[1]
    suffix = sys.argv[2]

    num_fixed_arg = 2
    num_loss = len(sys.argv) - num_fixed_arg - 1
    loss_names = []
    for i in range(num_loss):
        arg_id = i + num_fixed_arg + 1
        loss_name = sys.argv[arg_id]
        loss_names.append(loss_name)

    cwd = os.getcwd()
    path_1 = os.path.join(cwd, subdir)
    file_paths = glob.glob(path_1 + '/*.' + suffix)
    if len(file_paths) == 0:
        raise Exception('Not found log file.')
    else:
        show_curves_multi_file(file_paths, loss_names)