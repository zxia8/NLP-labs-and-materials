# This script is created by Ziyuan
# Edited and compiled under Windows OS
# Implemented text classification

import os
import argparse
import re
from collections import Counter
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt


class CommandLine:
    # Process command line args
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("folder_name")
        args = parser.parse_args()

        self.folder_name = args.folder_name


class Comment:
    # define a class of comments, which has the comment's BOW (or bi-gram or trigram) model
    # and its polarity (+ve or -ve)
    def __init__(self, model, polarity):
        self.model = model
        self.polarity = polarity
        if polarity == 'pos':
            self.y = 1
        else:
            self.y = -1


def get_list(folder_name, gram, polarity):
    # travers the folder, get files.
    file_content_train = list()
    file_content_test = list()
    for root, dirs, files in os.walk(folder_name):
        # read file content, create BOW, save as Comment object, push to train and test list
        for name in files:
            f = open(os.path.join(root, name), 'r')
            file_content = re.sub("[^\w']", " ", f.read().lower()).split()[:]
            if gram == 'tri':
                file_content = list(zip(file_content, file_content[1:], file_content[2:]))
            elif gram == 'bi':
                file_content = list(zip(file_content, file_content[1:]))
            model = Counter(file_content)
            comment = Comment(model, polarity)
            if len(file_content_train) <= 799:
                file_content_train.append(comment)
            else:
                file_content_test.append(comment)
            f.close()
    return file_content_train, file_content_test


def train(comment_list, test_list, model):
    a_list = list()
    f_list = list()
    pass_list = list()
    # train parameter in this function

    # create weight dictionary w and storing dictionary w_store
    w = dict()
    w_store = dict()
    c = 1
    for comment in comment_list:
        for key in comment.model:
            w[key] = 0
            w_store[key] = 0

    # start training with multiple passes (say 100 passes)
    for i in range(100):
        shuffle(comment_list)
        for comment in comment_list:
            a = 0
            # compute w*phi
            for key in comment.model:
                a += w[key]*comment.model[key]
            # make prediction
            y_hat = np.sign(a)
            # learning weight
            for key in comment.model:
                if y_hat != comment.y:
                    w[key] += comment.y*comment.model[key]
        c += 1
        # store all previous w in w_store
        for key in w:
                    w_store[key] += w[key]

        # get features for plotting
        t_dic = test(w_store, test_list)
        a_list.append(t_dic['accuracy'])
        f_list.append(t_dic['f1'])
        pass_list.append(i)
    # plot the training progress by pass, save to file
    plt.figure()
    plt.plot(pass_list, a_list, label='accuracy')
    plt.plot(pass_list, f_list, color='red', linewidth=1.0, linestyle='--', label='f1 measure')
    plt.xlabel('pass number')
    plt.title('accuracy vs. f1')
    plt.legend()
    plt.savefig(model)
    plt.close()

    # compute the average of all w as training result
    for key in w_store:
        w_store[key] /= c

    return w_store


def get_top_ten(dic):
    # obtain the top 10 positively/negatively weighted features
    sd = sorted(dic.items(), key=lambda item: item[1])
    top_ten_pos = list()
    top_ten_neg = list()
    for i in range(10):
        top_ten_pos.append(sd.pop())
        top_ten_neg.append(sd.pop(0))
    return top_ten_pos, top_ten_neg


def test(w_dic, comment_list):
    true_p = 0
    true_n = 0
    false_p = 0
    false_n = 0

    # shuffle the test list, not necessary, personal preference
    shuffle(comment_list)
    for comment in comment_list:
        a = 0
        # compute w*phi
        for key in comment.model:
            if key in w_dic:
                a += w_dic[key]*comment.model[key]
                # print(a)
        # make prediction
        y_hat = np.sign(a)

        # count features for computing evaluation components
        if (y_hat == comment.y) & (y_hat == 1):
            true_p += 1
        if (y_hat != comment.y) & (y_hat == 1):
            false_p += 1
        if (y_hat != comment.y) & (y_hat == -1):
            false_n += 1
        if (y_hat == comment.y) & (y_hat == -1):
            true_n += 1
    # compute accuracy, precision, recall and f_1 measure for evaluation
    accuracy = (true_p+true_n)/(true_p+true_n+false_p+false_n)
    precision = true_p/(true_p+false_p)
    recall = true_p/(true_p+false_n)
    f1 = 2*precision*recall/(precision+recall)

    return {'accuracy': accuracy, 'precision': precision,
            'recall': recall, 'f1': f1}


def show_result(model, result, dic):
    # print all result in a personal preferred way
    print('---------------------------------------------------')
    print('---------------------------------------------------')
    print('The evaluation of ' + model + ' is: ')
    print(result)
    print()
    print('the most positively weighted features for ' + model + ' model is: ')
    print(get_top_ten(dic)[0])
    print()
    print('the most negatively weighted features for ' + model + ' model is: ')
    print(get_top_ten(dic)[1])
    print()
    print('training process graph saved in this dir')


if __name__ == '__main__':
    config = CommandLine()
    folder_name = config.folder_name + '/text_sentoken'
    # folder_name = 'review_polarity'

    print('building models... ...')
    neg_uni_train, neg_uni_test = get_list(folder_name+'/neg', 'uni', 'neg')
    neg_bi_train, neg_bi_test = get_list(folder_name+'/neg', 'bi', 'neg')
    neg_tri_train, neg_tri_test = get_list(folder_name+'/neg', 'tri', 'neg')
    pos_uni_train, pos_uni_test = get_list(folder_name+'/pos', 'uni', 'pos')
    pos_bi_train, pos_bi_test = get_list(folder_name+'/pos', 'bi', 'pos')
    pos_tri_train, pos_tri_test = get_list(folder_name+'/pos', 'tri', 'pos')
    print('complete building model')

    print('training unigram model... ...')
    print()
    weight_dic_uni = train(pos_uni_train + neg_uni_train, pos_uni_test+neg_uni_test, 'unigram')
    uni_result = test(weight_dic_uni, pos_uni_test+neg_uni_test)
    show_result('unigram', uni_result, weight_dic_uni)

    print('now start training bigram model... ...')
    weight_dic_bi = train(pos_bi_train + neg_bi_train, pos_bi_test + neg_bi_test, 'bigram')
    bi_result = test(weight_dic_bi, pos_bi_test + neg_bi_test)
    show_result('bigram', bi_result, weight_dic_bi)

    print('now start training trigram model... ...')
    weight_dic_tri = train(pos_tri_train + neg_tri_train, pos_tri_test + neg_tri_test, 'trigram')
    tri_result = test(weight_dic_tri, pos_tri_test + neg_tri_test)
    show_result('trigram', tri_result, weight_dic_tri)





