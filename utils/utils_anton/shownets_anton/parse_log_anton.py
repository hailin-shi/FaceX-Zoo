import re


ptrn_epoch = re.compile('Epoch (\d+)')
ptrn_iter = re.compile('iter (\d+)/(\d+)')








def parse_log(log_path, loss_name):
    ptrn_loss = re.compile(loss_name + ' ([-+]?[0-9]*\.?[0-9]+)')
    curve_dict = {}
    with open(log_path) as f:
        iter_long = 0
        for line in f:
            r = ptrn_loss.search(line)
            if r:
                meta_dict = {}
                epoch = int(ptrn_epoch.search(line).groups()[0])
                iter_curr, iter_pd = int(ptrn_iter.search(line).groups()[0]), int(ptrn_iter.search(line).groups()[1])
                loss_value = float(ptrn_loss.search(line).groups()[0])
                meta_dict['epoch'] = epoch
                meta_dict['iter_curr'] = iter_curr
                meta_dict['iter_pd'] = iter_pd
                meta_dict[loss_name] = loss_value

                curve_dict[iter_long] = meta_dict
                iter_long = iter_long + 1

    return curve_dict