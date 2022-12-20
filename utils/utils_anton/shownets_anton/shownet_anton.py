from parse_log_anton import *
import sys
import os
from matplotlib import  pyplot as plt

colors = ['b', 'g', 'c', 'm', 'y', 'k'] * 10


def show_curve(log_path, loss_name):

    if not os.path.exists(log_path):
        raise Exception("The log file does not exist or the path is not correct!")

    curv_dict = parse_log(log_path, loss_name)
    iter_list = list(curv_dict.keys())
    iter_list.sort()

    draw_dict ={}
    draw_dict['x_train'] = iter_list
    draw_dict['y_train'] = []
    for iter in iter_list:
        loss_value = curv_dict[iter][loss_name]
        draw_dict['y_train'].append(loss_value)

    min_y = min(draw_dict['y_train'])
    max_y = max(draw_dict['y_train'])


    figure = plt.figure(figsize=[9, 3])
    figure.add_subplot(1, 1, 1)
    plt.ylim([min_y, max_y])
    plt.xlabel('iteration * 20')
    plt.ylabel(loss_name)
    plt.plot(draw_dict['x_train'], draw_dict['y_train'], colors[0], linewidth=2)
    plt.title(loss_name + " vs. iterations")
    plt.legend(loc='best')
    plt.grid()

    plt.show()


if __name__ == "__main__":
    log_path = sys.argv[1]
    loss_name = sys.argv[2]

    show_curve(log_path, loss_name)






