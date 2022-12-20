#!/usr/bin/env python
# coding: utf-8

import re


regex_iteration = re.compile('Iteration (\d+)')

pattern_train_output = re.compile('Train net output #(\d+): (.*?) = ([-+]?[0-9]*\.?[0-9]+)')
pattern_test_output = re.compile('Test net output #(\d+): (.*?) = ([-+]?[0-9]*\.?[0-9]+)')


def init_dicts(loss_names):
    dict_train,  dict_test = {}, {}
    for loss_name in loss_names:
        dict_train[loss_name] = {}
        dict_test[loss_name] = {}

    return dict_train, dict_test


def parse_log(path_to_log, dict_train, dict_test, loss_names):

    current_iter = -1
    iters = []


    with open(path_to_log) as f:

        for line in f:

            # update iteration
            iteration_match = regex_iteration.search(line)
            if iteration_match:
                current_iter = float(iteration_match.group(1))
            if current_iter == -1:
                continue

            if not current_iter in iters:
                iters.append(current_iter)

            # parse train
            r = pattern_train_output.search(line)
            if r:
                _, loss_name, loss_value = r.groups()
                if loss_name in loss_names:
                    dict_train[loss_name][current_iter] = float(loss_value)

                continue

            # parse test
            r = pattern_test_output.search(line)
            if r:
                _, loss_name, loss_value = r.groups()
                if loss_name in loss_names:
                    dict_test[loss_name][current_iter] = float(loss_value)

                continue

    return dict_train, dict_test, iters





