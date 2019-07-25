# This script is created by Ziyuan
# Edited and compiled under Windows OS
# Implemented named entity recognition
import argparse
import random
import re
from itertools import product
from pprint import pprint
from sklearn.metrics import f1_score


class CommandLine:
    # Process command line args
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("train_file")
        parser.add_argument("test_file")
        args = parser.parse_args()

        self.train_file = args.train_file
        self.test_file = args.test_file


def load_dataset_sents(file_path, as_zip=True, to_idx=False, token_vocab=None, target_vocab=None):
    targets = []
    inputs = []
    zip_inps = []
    with open(file_path) as f:
        for line in f:
            sent, tags = line.split('\t')
            words = [token_vocab[w.strip()] if to_idx else w.strip() for w in sent.split()]
            ner_tags = [target_vocab[w.strip()] if to_idx else w.strip() for w in tags.split()]
            inputs.append(words)
            targets.append(ner_tags)
            zip_inps.append(list(zip(words, ner_tags)))
    return zip_inps if as_zip else (inputs, targets)


def cw_cl_count(c):
    """
    get current word - current label count
    :param c: input a loaded data set
    :return: a dictionary with data in keys and count in values, low freq words is cut off
    """
    cw_cl_temp = dict()
    cw_cl_dict = dict()
    cut_off_frequency = 1
    for s in c:
        for t in s:
            token = t[0] + '_' + t[1]
            if token in cw_cl_temp.keys():
                cw_cl_temp[token] += 1
            else:
                cw_cl_temp[token] = 1
    for k in cw_cl_temp:
        if cw_cl_temp[k] >= cut_off_frequency:
            cw_cl_dict[k] = cw_cl_temp[k]
    return cw_cl_dict


def get_word_list(s):
    """
    change a sentence from loaded data into a word list
    :param s: a sentence from loaded data
    :return: a list which has all words from the sentence
    """
    word_list = list()
    for t in s:
        word_list.append(t[0])
    return word_list


def get_tag_list(s):
    """
    change a sentence from loaded data into a label list
    :param s: a sentence from loaded data
    :return: a list which has all tags from the sentence
    """
    tag_list = list()
    for t in s:
        tag_list.append(t[1])
    return tag_list


def phi_1(x, y, cw_cl_counts):
    """
    form phi_1
    :param x: a list of word
    :param y: a list of tag
    :param cw_cl_counts: current word current label dictionary, cutting off low freq words
    :return: a dictionary contains word_tag in keys and count of word_tag from sentence in values
    """
    phi_1_dict = dict()
    for i in range(len(x)):
        token = x[i] + '_' + y[i]
        # for keeping overlap word in the dictionary, multiple chr(3) is added
        if token in phi_1_dict:
            token = chr(3)*i + token
        if token in cw_cl_counts:
            if token in phi_1_dict:
                phi_1_dict[token] += 1
            else:
                phi_1_dict[token] = 1
        else:
            phi_1_dict[token] = 0
    return phi_1_dict


def pl_cl_count(c):
    """
    get previous label - current label count
    :param c: input a loaded data set
    :return: a dictionary with data in keys and count in values
    """
    pl_cl_dict = dict()
    tag_list = list()
    for s in c:
        tags = get_tag_list(s)
        tags.insert(0, 'None')
        tags = list(zip(tags[:], tags[1:]))
        for i in range(len(tags)):
            tags[i] = tags[i][0] + '_' + tags[i][1]
        tag_list += tags
    for t in tag_list:
        if t in pl_cl_dict:
            pl_cl_dict[t] += 1
        else:
            pl_cl_dict[t] = 1
    return pl_cl_dict


def phi_2(x, pl_cl_counts):
    """
        form phi_2
        :param x: a list of word
        :param pl_cl_counts: previous label current label dictionary
        :return: a dictionary contains tag_tag in keys and count of tag_tag from sentence in values
        """
    phi_2_dict = dict()
    x.insert(0, 'None')
    previous_label = x
    current_label = x
    for i in range(len(previous_label)-1):
        pl_cl = previous_label[i] + '_' + current_label[i+1]
        # for keeping overlap keys in the dictionary, multiple chr(3) is added
        if pl_cl in phi_2_dict:
            pl_cl = chr(3)*i + pl_cl
        if pl_cl in pl_cl_counts:
            if pl_cl in phi_2_dict:
                phi_2_dict[pl_cl] += 1
            else:
                phi_2_dict[pl_cl] = 1
        else:
            phi_2_dict[pl_cl] = 0
    return phi_2_dict


def train(train_data, cw_cl_count, pl_cl_count):
    """
    train the structured perceptron with phi_1 (pl_cl_count is None) and phi_1 + phi_2 (pl_cl_count exists)
    :param train_data: loaded data from train file
    :param cw_cl_count: current word current label dictionary
    :param pl_cl_count: previous label current label dictionary
    :return: trained weight with randomising and averaging
    """
    tags = ['O', 'PER', 'LOC', 'ORG', 'MISC']
    # initialising
    w = dict()
    phi_store = list()
    for i in range(len(train_data)):
        word = get_word_list(train_data[i])
        tag = get_tag_list(train_data[i])
        tag_order = product(tags, repeat=len(train_data[i]))
        phi_sentence = list()
        for item in tag_order:
            if pl_cl_count == None:
                phi_sentence.append(phi_1(word, item, cw_cl_count))
            else:
                phi_c = dict()
                phi_c.update(phi_1(word, item, cw_cl_count))
                phi_c.update(phi_2(list(item), pl_cl_count))
                phi_sentence.append(phi_c)
            for j in range(len(train_data[i])):
                w_l = word[j] + '_' + item[j]
                w[w_l] = 0
                if pl_cl_count != None:
                    l_l = item[j] + '_' + tag[j]
                    w[l_l] = 0
        phi_store.append(phi_sentence)
    combined_data = list()
    for i in range(len(train_data)):
        combined_data.append((train_data[i], phi_store[i]))
    # initialise w store
    w_store = dict()
    for key in w.keys():
        w_store[key] = [0, 0]
    # start training with multiple passes
    for ps in range(3):
        print(ps + 1, 'train iteration')
        random.seed(ps+10)
        random.shuffle(combined_data)
        random_train_data = list()
        random_phi_store = list()
        # separate shuffled data and phi
        for index in range(len(train_data)):
            random_train_data.append(combined_data[index][0])
            random_phi_store.append(combined_data[index][1])
        for i in range(len(random_train_data)):
            word = get_word_list(random_train_data[i])
            true = get_tag_list(random_train_data[i])
            predicted = predict(w, random_phi_store[i])
            phi_dic = dict()
            for k in range(len(random_phi_store[i])):
                phi_dic.update(random_phi_store[i][k])
            # updating w
            if predicted[:len(true)] != true:
                for j in range(len(word)):
                    true_tag = word[j] + '_' + true[j]
                    predicted_tag = word[j] + '_' + predicted[j]
                    w[true_tag] += phi_dic[true_tag] - phi_dic[predicted_tag]
                    w_store[true_tag][0] += w[true_tag]
                    w_store[true_tag][1] += 1
                # with phi_2 included, updating tag_tag weight
                if pl_cl_count != None:
                    true.insert(0, 'None')
                    predicted.insert(0, 'None')
                    for j in range(len(word)):
                        true_tag = true[j] + '_' + true[j+1]
                        predicted_tag = predicted[j] + '_' + predicted[j+1]
                        w[true_tag] += phi_dic[true_tag] - phi_dic[predicted_tag]
                        if true_tag in w_store:
                            w_store[true_tag][0] += w[true_tag]
                            w_store[true_tag][1] += 1
                        else:
                            w_store[true_tag] = [w[true_tag], 1]
    # compute average weight
    for key in w_store:
        if w_store[key][1] != 0:
            w_store[key] = w_store[key][0]/w_store[key][1]
        else:
            w_store[key] = w_store[key][0]
    return w_store


def predict(w, phi):
    """
    predict the most likely tag sequence of a sentence
    :param w: current weight
    :param phi: feature set
    :return: predicted tag sequence
    """
    predicted_tag = list()
    highest_score = -100
    bug = re.compile(chr(3))
    for phi_s in phi:
        sentence_score = 0
        sentence_tag = list()
        for key, value in phi_s.items():
            # cancel the chr(3) added in phi function to find the score
            if bug.match(key):
                key = key.split(chr(3))[-1]
            if key not in w:
                w[key] = 0
            sentence_score += w[key]*value
            sentence_tag.append(key.split('_')[1])
        if sentence_score > highest_score:
            highest_score = sentence_score
            predicted_tag = sentence_tag
    return predicted_tag


def test(w, test_data, cw_cl_count, pl_cl_count):
    """
    run the perceptron on test data set to measure the performance
    :param w: the trained weights
    :param test_data: loaded test data
    :param cw_cl_count: current word current label dictionary
    :param pl_cl_count: previous label current label dictionary
    :return: F1 score of the perceptron
    """
    tags = ['O', 'PER', 'LOC', 'ORG', 'MISC']
    phi_store = list()
    for i in range(len(test_data)):
        word = get_word_list(test_data[i])
        tag_order = product(tags, repeat=len(test_data[i]))
        phi_sentence = list()
        for item in tag_order:
            if pl_cl_count == None:
                # phi_1 feature
                phi_sentence.append(phi_1(word, item, cw_cl_count))
            else:
                # phi_1_2 feature
                phi_c = dict()
                phi_c.update(phi_1(word, item, cw_cl_count))
                phi_c.update(phi_2(list(item), pl_cl_count))
                phi_sentence.append(phi_c)
        phi_store.append(phi_sentence)
    y_true = list()
    y_predicted = list()
    for i in range(len(test_data)):
        y_true += get_tag_list(test_data[i])
        y_predicted += predict(w, phi_store[i])[:len(get_tag_list(test_data[i]))]
    f1_micro = f1_score(y_true, y_predicted, average='micro', labels=['ORG', 'MISC', 'PER', 'LOC'])
    return f1_micro


def get_word_dic(d):
    """
    a step to get top 10 weights for each class, change keys xxx_xxx into (xxx, xxx)
    :param d: the trained weight dictionary
    :return: the trained weight dictionary with modified keys
    """
    word_dic = dict()
    for key in d:
        word_dic[(key.split('_')[0], key.split('_')[1])] = d[key]
    return word_dic


def seperate_dic(d):
    """
    separate weight dictionary into classes and get top ten weights
    :param d: the trained weight dictionary with modified keys
    :return: top ten weights for each class
    """
    dic_ORG = dict()
    dic_PER = dict()
    dic_MISC = dict()
    dic_LOC = dict()

    for key in d:
        if key[1] == 'ORG':
            dic_ORG[key[0]] = d[key]
        elif key[1] == 'PER':
            dic_PER[key[0]] = d[key]
        elif key[1] == 'MISC':
            dic_MISC[key[0]] = d[key]
        elif key[1] == 'LOC':
            dic_LOC[key[0]] = d[key]

    top_ORG = sorted(dic_ORG.items(), key=lambda item: item[1], reverse=True)[:10]
    top_PER = sorted(dic_PER.items(), key=lambda item: item[1], reverse=True)[:10]
    top_MISC = sorted(dic_MISC.items(), key=lambda item: item[1], reverse=True)[:10]
    top_LOC = sorted(dic_LOC.items(), key=lambda item: item[1], reverse=True)[:10]

    return top_ORG, top_PER, top_MISC, top_LOC


if __name__ == '__main__':
    config = CommandLine()
    print('initialising')
    train_file = config.train_file
    test_file = config.test_file
    train_data = load_dataset_sents(train_file)
    test_data = load_dataset_sents(test_file)
    cw_cl = cw_cl_count(train_data)
    pl_cl = pl_cl_count(train_data)
    print('start training phi_1')
    train_phi_1 = train(train_data, cw_cl, None)
    b = test(train_phi_1, test_data, cw_cl, None)
    print('f1 score of phi_1 is: ', b)
    print('-------------------------------------------')
    print('start training phi_1 + phi_2')
    train_phi_1_2 = train(train_data, cw_cl, pl_cl)
    c = test(train_phi_1_2, test_data, cw_cl, pl_cl)
    print('f1 score of phi_1 + phi_2 is: ', c)
    print('-------------------------------------------')

    word_phi_1 = get_word_dic(train_phi_1)
    word_phi_1_2_temp = get_word_dic(train_phi_1_2)
    word_phi_1_2 = dict()
    # remove tag_tag keys in phi_1_2
    for key in word_phi_1:
        word_phi_1_2[key] = word_phi_1_2_temp[key]
    top_ten_phi_1 = seperate_dic(word_phi_1)
    top_ten_phi_1_2 = seperate_dic(word_phi_1_2)
    print('top 10 of ORG in phi_1 is: ')
    pprint(top_ten_phi_1[0])
    print('top 10 of PER in phi_1 is: ')
    pprint(top_ten_phi_1[1])
    print('top 10 of MISC in phi_1 is: ')
    pprint(top_ten_phi_1[2])
    print('top 10 of LOC in phi_1 is: ')
    pprint(top_ten_phi_1[3])
    print('--------------------------------------------')
    print('top 10 of ORG in phi_1_2 is: ')
    pprint(top_ten_phi_1_2[0])
    print('top 10 of PER in phi_1_2 is: ')
    pprint(top_ten_phi_1_2[1])
    print('top 10 of MISC in phi_1_2 is: ')
    pprint(top_ten_phi_1_2[2])
    print('top 10 of LOC in phi_1_2 is: ')
    pprint(top_ten_phi_1_2[3])




