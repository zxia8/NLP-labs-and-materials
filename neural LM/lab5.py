# This script is created by Ziyuan
# Edited and compiled under Windows OS
# Implemented neural language model
"""
input layer (embedding): ctx size * embed size
hidden layer: 128 dim
output layer: bow length
"""
import pickle
import random
from random import shuffle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


torch.manual_seed(0)

######################################################################

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

#                     Create Training Set                            #
######################################################################
sentence_1 = "<s> The mathematician ran .".split()
sentence_2 = "<s> The mathematician ran to the store .".split()
sentence_3 = "<s> The physicist ran to the store .".split()
sentence_4 = "<s> The philosopher thought about it .".split()
sentence_5 = "<s> The mathematician solved the open problem .".split()
train_set = [sentence_1, sentence_2, sentence_3, sentence_4, sentence_5]
######################################################################


trigrams = []
trigrams_separated = []
vocab_ = []
for sentence in train_set:
    trigrams += [([sentence[i], sentence[i + 1]], sentence[i + 2])for i in range(len(sentence) - 2)]
    trigrams_separated.append([([sentence[i], sentence[i + 1]], sentence[i + 2])for i in range(len(sentence) - 2)])
    vocab_ += sentence
random.seed(7849347351)
vocab = sorted(set(vocab_))
shuffle(vocab)
# vocab = set(vocab_)  # <<<< <<<< activate this command to run sanity check <<<<  <<<<   <<<<
word_to_ix = {word: i for i, word in enumerate(vocab)}


class NGramLanguageModeler(nn.Module):

    def __init__(self, vocab_size, embedding_dim, context_size):
        super(NGramLanguageModeler, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs


def predict(context, v):
    """
    Given a context with length of 2, this function predicts next token
    :param context: context with length of 2, inputted as a list
    :param v: word to index dictionary
    :return: the predicted token (word)
    """
    c = [v[context[0]], v[context[1]]]  # invert to index
    var = autograd.Variable(torch.LongTensor(c))
    idx = torch.argmax(model(var)).item()  # predict
    rst = 0
    for k in v:
        if v[k] == idx:
            rst = k
    return rst


def check_every_trigram(trigrams, v):
    """
    Given all trigrams, this function computes the accuracy of prediction
    :param trigrams: a list of all trigrams from the model
    :param v: word to index dictionary
    :return: the accuracy of prediction (five-digit precision)
    """
    ctx_list = []  # stores context
    tar_list = []  # stores target
    c_true = 0
    c_false = 0
    for key in trigrams:
        ctx_list.append(key[0])
        tar_list.append(key[1])

    for i in range(len(ctx_list)):
        if predict(ctx_list[i], v) != tar_list[i]:
            c_false += 1
        else:
            c_true += 1
    return round(c_true/(c_true + c_false), 5)


def compute_log_prob(trigram, v):
    """
    Given a sentence (in trigram form), returns its' log_prob.sum()
    :param trigram: a sentence in trigram form: [([con,con],tar), ...]
    :param v: word to index dictionary
    :return: the added up log probability of the input sentence
    """
    ctx_list = []
    tar_list = []
    log_prob = 0
    for k in trigram:  # init context and target list
        ctx_list.append(k[0])
        tar_list.append(k[1])
    for i in range(len(ctx_list)):  # accumulate log probabilities
        ctx = [v[ctx_list[i][0]], v[ctx_list[i][1]]]
        var = autograd.Variable(torch.LongTensor(ctx))
        tar_idx = v[tar_list[i]]
        log_prob += model(var)[0][tar_idx]
    return log_prob


def sanity_check(list, v):
    if check_every_trigram(list, v) == 1:  # sanity check here <<<<

        print("####sanity check returns success!     ####    ####       "
              "The sentence has been successfully returned!############")
        print("                                    ################# ")
        print("                                  #################### ")
        print("                                   ################### ")
        print("                                     ###############  ")
        print("                                       ###########  ")
        print("                                         ###### ")
        print("                                           ##  ")


losses = []
loss_function = nn.NLLLoss()

"""
embedding dim is set to |vocab|^0.25
"""
EMBEDDING_DIM = round(len(vocab)**0.25)
model = NGramLanguageModeler(len(vocab), EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.SGD(model.parameters(), lr=0.03)


#                                        training                                                       ##
##########################################################################################################
vector = []  # used to make graph not useful in this implementation
for epoch in range(85):
    total_loss = torch.Tensor([0])

    for context, target in trigrams:

        # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
        # into integer indices and wrap them in variables)
        context_idxs = [word_to_ix[w] for w in context]
        context_var = autograd.Variable(torch.LongTensor(context_idxs))
        # print(context_idxs)

        # Step 2. Recall that torch *accumulates* gradients. Before passing in a
        # new instance, you need to zero out the gradients from the old
        # instance
        model.zero_grad()

        # Step 3. Run the forward pass, getting log probabilities over next
        # words
        log_probs = model(context_var)

        # print(log_probs)
        # print(context_var)

        # Step 4. Compute your loss function. (Again, Torch wants the target
        # word wrapped in a variable)
        loss = loss_function(log_probs, autograd.Variable(
            torch.LongTensor([word_to_ix[target]])))

        # Step 5. Do the backward pass and update the gradient
        loss.backward()
        optimizer.step()

        total_loss += loss.data
    accuracy = check_every_trigram(trigrams, word_to_ix)
    # vector.append(accuracy)

    if (epoch+1) % 5 == 0 or epoch <= 4:
        ctxt = ['<s>', 'The']
        result = predict(ctxt, word_to_ix)
        print("The predicted target in the context of ['<s>', 'The'] of epoch", epoch, 'is: ', result)
        print("The accuracy of prediction for every trigram of epoch", epoch, 'is: ',
              accuracy)
        print('-------------------------------------------------------------------------------------------------------')
    losses.append(total_loss)

# with open('acc_info_3', 'wb') as f:
#     pickle.dump(vector, f)
# f.close()

sanity_check(trigrams_separated[1], word_to_ix)
print('current loss:', losses[-1].item())
#                                            testing                                                     #
##########################################################################################################
print("##############################################################################################")
print("#                           test with 'solve open problem sentence'                          #")
print("#                                                                                            #")
test_sentence_1 = "<s> The philosopher solved the open problem .".split()
trigram_1 = [([test_sentence_1[i], test_sentence_1[i + 1]], test_sentence_1[i + 2])
             for i in range(len(test_sentence_1) - 2)]
test_sentence_2 = "<s> The physicist solved the open problem .".split()
trigram_2 = [([test_sentence_2[i], test_sentence_2[i + 1]], test_sentence_2[i + 2])
             for i in range(len(test_sentence_2) - 2)]
trigram_test = [trigram_1, trigram_2]
# print(trigram_test)
score_phi = compute_log_prob(trigram_1, word_to_ix)
score_phy = compute_log_prob(trigram_2, word_to_ix)
if score_phy > score_phi:
    print("# The model predicts that 'physicist' is the correct answer                                  #")
else:
    print("# The model predicts that 'philosopher' is the correct answer                                #")
print("##############################################################################################")
print('')
print('')

#                                        cosine similarity                                               #
##########################################################################################################
print("##############################################################################################")
print("#                                 display cosine similarity                                  #")
print("##############################################################################################")
phy_var = autograd.Variable(torch.LongTensor([word_to_ix["physicist"]]))
phi_var = autograd.Variable(torch.LongTensor([word_to_ix["philosopher"]]))
math_var = autograd.Variable(torch.LongTensor([word_to_ix["mathematician"]]))
phy_math = F.cosine_similarity(model.embeddings(phy_var).view((1, -1)),
                               model.embeddings(math_var).view((1, -1)))
phi_math = F.cosine_similarity(model.embeddings(phi_var).view((1, -1)),
                               model.embeddings(math_var).view((1, -1)))
print("The cos similarity between physicist and mathematician is:", round(phy_math.item(), 5))
print("The cos similarity between philosopher and mathematician is:", round(phi_math.item(), 5))
if phy_math > phi_math:
    print("Physicist is more similar with mathematician than philosopher")
else:
    print("Philosopher is more similar with mathematician than physicist")












