# Created by Ziyuan
# Edited and compiled under Windows OS
# Implemented language modeling

import argparse
import re
import string
from pprint import pprint


class CommandLine:
    # Process command line args
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("train_file")
        parser.add_argument("question_file")
        args = parser.parse_args()

        self.train_file = args.train_file
        self.question_file = args.question_file


def uni_model(file):
    # read file and build unigram model in this function
    f = open(file, 'r', encoding="utf8")
    content_list = f.readlines()
    f.close()
    token_count = dict()
    count_uni = 0
    # a dictionary to store probabilities
    p_dic = dict()
    # create model with tokenization and capitalisation
    for i in range(len(content_list)):
        content_list[i] = content_list[i].lower().split()
        # print(content_list[i])
        # pre-compile continuous punctuation (for removing '--', '???', etc.)
        repeat_punc = re.compile(r'(([^\w\s])\2+)')
        for w in content_list[i]:
            if w in string.punctuation or repeat_punc.match(w):
                content_list[i].remove(w)
        # insert start and end of sentence symbol
        content_list[i].insert(0, '<s>')
        content_list[i].append('</s>')
        # count tokens
        for w in content_list[i]:
            if w in token_count:
                token_count[w] += 1
            else:
                token_count[w] = 1
        count_uni += len(content_list[i])
    # compute probability
    for k in token_count:
            p_dic[k] = token_count[k]/count_uni
    # content_list will be used to build bigram model
    # token_count will be use as bigram model's context count
    # p_dic is the unigram model
    return content_list, token_count, p_dic


def bi_model(smooth, *args):
    content_list = args[0]
    ct_list = list()
    token_count = args[1]
    token_count_bi = dict()
    count_bi = 0

    for i in range(len(content_list)):
        # zip every two elements from unigram model to form element of bigram model
        ct_list.append(list(zip(content_list[i], content_list[i][1:])))
        count_bi += len(ct_list[i])
        # count bigram tokens
        for w in ct_list[i]:
            if w in token_count_bi:
                token_count_bi[w] += 1
            else:
                token_count_bi[w] = 1
    if smooth:
        # calculate probability with add-1 smoothing
        prob_dict = prob(token_count_bi, 'bigram', token_count, count_bi, 'smooth')
    else:
        # calculate probability without smoothing
        prob_dict = prob(token_count_bi, 'bigram', token_count)
    # prob_dict is bigram model
    # token_count and count_bi will be used in solving problem with smoothing session
    return prob_dict, token_count, count_bi


def prob(dic, model_type, *args):
    # compute probability for bigram models
    d = dict()
    for key in dic:
        if model_type == 'bigram' and 'smooth' not in args:
            d[key] = dic[key]/args[0][key[0]]
        elif model_type == 'bigram' and 'smooth' in args:
            d[key] = (dic[key] + 1) / (args[0][key[0]] + args[1])
    return d


def questions(file):
    # read question file and convert to suitable format: {index: (modeled_question, choices)}
    f = open(file, 'r', encoding="utf8").readlines()
    q_dict = dict()
    i = 1
    for l in f:
        question, choice = l.split(':')
        question = question.lower().split()
        for w in question:
            if w in string.punctuation:
                question.remove(w)
        question.insert(0, '<s>')
        question.append('</s>')
        choice = choice.strip().split('/')
        q_dict[i] = (question, choice)
        i += 1
    return q_dict


def solve_questions(model_t, q_dic, p_dic, *args):
    # use language model to determine which choice is correct
    answers = list()
    for k in q_dic:
        answer = ''
        index = get_index(q_dic[k][0])
        if model_t == 'unigram':
            token_1 = q_dic[k][1][0]
            token_2 = q_dic[k][1][1]
            # find unigram probability
            w_1 = find_prob(token_1, p_dic)
            w_2 = find_prob(token_2, p_dic)
        else:
            # compute conditional probability
            first_token_1 = (q_dic[k][0][index - 1], q_dic[k][1][0])
            first_token_2 = (q_dic[k][0][index - 1], q_dic[k][1][1])
            second_token_1 = (q_dic[k][1][0], q_dic[k][0][index + 1])
            second_token_2 = (q_dic[k][1][1], q_dic[k][0][index + 1])
            if 'smooth' in args:
                first_p_1 = find_prob(first_token_1, p_dic, args[0], args[1], 'smooth')
                first_p_2 = find_prob(first_token_2, p_dic, args[0], args[1], 'smooth')
                second_p_1 = find_prob(second_token_1, p_dic, args[0], args[1], 'smooth')
                second_p_2 = find_prob(second_token_2, p_dic, args[0], args[1], 'smooth')
            else:
                first_p_1 = find_prob(first_token_1, p_dic, args[0])
                first_p_2 = find_prob(first_token_2, p_dic, args[0])
                second_p_1 = find_prob(second_token_1, p_dic, args[0])
                second_p_2 = find_prob(second_token_2, p_dic, args[0])
            # compute the probability of the sentence
            # as every other words are the same, only different part are considered
            w_1 = first_p_1*second_p_1
            w_2 = first_p_2*second_p_2
        # choose and generate answer (argmax w)
        if w_1 > w_2:
            q_dic[k][0][index] = q_dic[k][1][0]
            for w in q_dic[k][0]:
                answer += w + ' '
        elif w_1 < w_2:
            q_dic[k][0][index] = q_dic[k][1][1]
            for w in q_dic[k][0]:
                answer += w + ' '
        elif w_1 == w_2 and w_1 != 0:
            answer = 'equal probability, cannot choose'
        elif w_1 == w_2 and w_1 == 0:
            answer = 'zero probability, cannot choose'
        answers.append(answer)

    return answers


def find_prob(token, p_dic, *args):
    # find the choices' probability in language models
    p = 0
    if token in p_dic:
        p = p_dic[token]
    elif token not in p_dic and 'smooth' in args:
        if token[0] in args[0]:
            p = 1/(args[0][token[0]] + args[1])
        else:
            p = 1/args[1]
    return p


def get_index(l):
    # find where the blank is
    blank = re.compile('____')
    for i in range(len(l)):
        if blank.match(l[i]):
            return i


if __name__ == '__main__':
    config = CommandLine()
    train_file = config.train_file
    question_file = config.question_file
    # train_file = "news-corpus-500k.txt"
    # question_file = "questions.txt"

    c_list, t_count, prob_dic_uni = uni_model(train_file)
    q = questions(question_file)
    result_uni = solve_questions('unigram', q, prob_dic_uni)
    pprint('The result of unigram model is: ')
    pprint(result_uni)
    print('----------------------------------------------------------------')

    prob_dic_bi_ns, context_dic_bi_ns = bi_model(False, c_list, t_count)[:2]
    q = questions(question_file)
    result_bi_ns = solve_questions('bigram', q, prob_dic_bi_ns, context_dic_bi_ns)
    pprint('The result of bigram model is: ')
    pprint(result_bi_ns)
    print('----------------------------------------------------------------')

    prob_dic_bi_s, context_dic_bi_s, mod_V = bi_model(True, c_list, t_count)
    q = questions(question_file)
    result_bi_s = solve_questions('bigram', q, prob_dic_bi_s, context_dic_bi_s, mod_V, 'smooth')
    pprint('The result of bigram model with add-1 smooth is: ')
    pprint(result_bi_s)





