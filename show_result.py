import argparse
from sys import platform
import numpy as np
import matplotlib.pyplot as plt
import os
from foundations import local

def plot(test_iter, test_loss, test_acc, filename):
    fig, axes = plt.subplots(2,1)
    
    axes[0].plot(test_iter, test_loss)
    axes[1].plot(test_iter, test_acc)

    # axes[1].set_ylim((0.9,1))
    plt.savefig(filename+'.png')
    plt.close()

    return None

def plot_item(item, path, type):
    fig, ax = plt.subplots()
    num = item.size
    x = np.arange(num)
    ax.plot(x, item)
    ax.set_title(type)
    plt.savefig(path+type+'.png')
    plt.close()
    return

def read_logger(filename):

    with open(filename, 'r+') as f:

        train_acc = []
        train_loss = []
        train_iter = []
        test_acc = []
        test_loss = []
        test_iter =[]

        while True:
            line = f.readline()
            if not line:
                break
            item = line.split(',')
            if item[0] == 'train_loss':
                train_loss.append(float(item[2]))
                train_iter.append(item[1])
            elif item[0] == 'train_accuracy':
                train_acc.append(float(item[2]))
            elif item[0] == 'test_loss':
                test_loss.append(float(item[2]))
                test_iter.append(int(item[1]))
            elif item[0] == 'test_accuracy':
                test_acc.append(float(item[2]))

        train_loss = np.array(train_loss)
        train_acc = np.array(train_acc)
        test_loss = np.array(test_loss)
        test_acc = np.array(test_acc)
        train_iter = np.array(train_iter)
        test_iter = np.array(test_iter)

    return train_iter, train_loss, train_acc, test_iter, test_loss, test_acc


def level_continuous_reader(filepath, file='main'):
    replicate = [i for i in os.listdir(filepath) if '.' not in i]
    replicate = ['replicate_1']
    last_loss, last_acc, best_acc, train_acc_last= [],[],[], []
    for num in replicate:
        replicate_path = os.path.join(filepath, num)
        level = os.listdir(replicate_path)
        _level = [i for i in level if 'pretrain' not in i and '.' not in i]
        _level.sort(key = lambda x: int(x[6:]))
        l = 0
        for dir in _level:
            level_path = os.path.join(replicate_path, dir)
            filename = os.path.join(level_path, file) +'/logger'
            
            if os.path.exists(filename):
                train_iter, train_loss, train_acc, test_iter, test_loss, test_acc = read_logger(filename)
                # plot(test_iter, test_loss, test_acc, filename+'test')
                # plot(train_iter, train_loss, train_acc, filename+'train')
                if l == 0:
                    loss = test_loss
                    acc = test_acc
                    iter = test_iter
                else:
                    loss =np.append(loss, test_loss)
                    acc = np.append(acc, test_acc)
                    iter = np.append(iter, test_iter+1+iter[-1])
                last_loss.append(test_loss[-1])
                last_acc.append(test_acc[-1])
                best_acc.append(np.max(test_acc))
                if len(train_acc) != 0:
                    train_acc_last.append(train_acc[-1])
            else:
                if l == 0:
                    filename_main = os.path.join(level_path, 'main') +'/logger'
                    train_iter, train_loss, train_acc, test_iter, test_loss, test_acc = read_logger(filename_main)
                    loss = test_loss
                    acc = test_acc
                    iter = test_iter

            l += 1
    loss = loss.reshape(len(replicate), -1)
    acc = acc.reshape(len(replicate), -1)
    iter = iter.reshape(len(replicate), -1)[0]
    loss = np.mean(loss, 0)
    acc = np.mean(acc, 0)

    last_loss = np.array(last_loss).reshape(len(replicate), -1)
    last_acc = np.array(last_acc).reshape(len(replicate), -1)
    best_acc = np.array(best_acc).reshape(len(replicate), -1)
    last_loss = np.mean(last_loss, 0)
    last_acc = np.mean(last_acc, 0)
    best_acc = np.mean(best_acc, 0)
    if len(train_acc_last) != 0:
        train_acc_last = np.array(train_acc_last).reshape(len(replicate), -1)
        train_acc_last = np.mean(train_acc_last, 0)
    print('\n')
    for i in range(last_acc.size):
        print('%.2f'%(last_acc[i]*100), end=' ')
    print('')
    for i in range(best_acc.size):
        print('%.2f'%(best_acc[i]*100), end=' ')
    print('')
    if len(train_acc_last) != 0:
        for i in range(train_acc_last.size):
            print('%.2f'%(train_acc_last[i]*100), end=' ')
        print('\n')
    fig_name = filepath+'/main_continuous-loss-acc' if 'main' in file else filepath+'/branch_continuous-loss-acc' 
    # plot(iter, loss, acc, fig_name)

def single_level_reader(filepath, file='main'):
    replicate = [i for i in os.listdir(filepath) if '.' not in i]
    replicate = ['replicate_1']
    last_loss, last_acc, best_acc, train_acc_last= [],[],[], []
    for num in replicate:
        replicate_path = os.path.join(filepath, num)
        filename = os.path.join(replicate_path, file) +'/logger'
            
        if os.path.exists(filename):
            train_iter, train_loss, train_acc, test_iter, test_loss, test_acc = read_logger(filename)

    test_loss = test_loss.reshape(len(replicate), -1)
    train_loss = train_loss.reshape(len(replicate), -1)
    test_acc = test_acc.reshape(len(replicate), -1)
    train_acc = train_acc.reshape(len(replicate), -1)
    # iter = iter.reshape(len(replicate), -1)[0]
    test_loss = np.mean(test_loss, 0)
    train_loss = np.mean(train_loss, 0)
    test_acc = np.mean(test_acc, 0)
    train_acc = np.mean(train_acc, 0)

    print('last train loss %.2f'%(train_loss[-1]*100))
    print('last train acc %.2f'%(train_acc[-1]*100))
    print('last test acc %.2f'%(test_acc[-1]*100))
    print('best test acc %.2f'%(np.max(test_acc)*100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type = str,
                        help='The name of file.')

    parser.add_argument('--oneshot', action='store_true', help='Output the oneshot branch results') 
    # parser.add_argument('--train', action='store_true', help='Output the training results') 
    # parser.add_argument('--prune', action='store_true', help='Output the pruning results') 

    args = parser.parse_args()
    platform = local.Platform()
    file_path = os.path.join(platform.root,args.name)
    if 'train' in args.name:
        file = 'main'
        single_level_reader(file_path, )
    elif args.oneshot:
        file = 'branch_oneshot_prune_7b99794732481fff503d5290ee43fb4b'
        level_continuous_reader(file_path, file)
    else:
        file = 'main'
        level_continuous_reader(file_path, file)