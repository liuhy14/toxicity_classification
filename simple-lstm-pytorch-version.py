#!/usr/bin/env python
# coding: utf-8


# # Preface

# This kernel is a PyTorch version of the [Simple LSTM kernel](https://www.kaggle.com/thousandvoices/simple-lstm). All credit for architecture and preprocessing goes to @thousandvoices.
# There is a lot of discussion whether Keras, PyTorch, Tensorflow or the CUDA C API is best. But specifically between the PyTorch and Keras version of the simple LSTM architecture, there are 2 clear advantages of PyTorch:
# - Speed. The PyTorch version runs about 20 minutes faster.
# - Determinism. The PyTorch version is fully deterministic. Especially when it gets harder to improve your score later in the competition, determinism is very important.
# 
# I was surprised to see that PyTorch is that much faster, so I'm not completely sure the steps taken are exactly the same. If you see any difference, we can discuss it in the comments :)
# 
# The most likely reason the score of this kernel is higher than the @thousandvoices version is that the optimizer is not reinitialized after every epoch and thus the parameter-specific learning rates of Adam are not discarded after every epoch. That is the only difference between the kernels that is intended.

# # Imports & Utility functions

# In[8]:


import numpy as np
import pandas as pd
import os
import time
import gc
import random
from tqdm._tqdm_notebook import tqdm_notebook as tqdm
from keras.preprocessing import text, sequence
import torch
from torch import nn
from torch.utils import data
from torch.nn import functional as F
from tensorboardX import SummaryWriter

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--clean_pre', type = bool, default = False,
                   help='clean_preprocess')
parser.add_argument('--delstop', type = bool, default = False,
                   help='delete stop words')
parser.add_argument('--epoch', type = int, default = 5,
                   help='epoches')
parser.add_argument('--lr', type = float, default = 0.001,
                   help='learning rate')
args = parser.parse_args()
print('using cuda', torch.cuda.is_available())

# In[ ]:


import nltk
nltk.download('stopwords')
nltk.download('punkt')
from stop_words import get_stop_words
from nltk.corpus import stopwords

stop_words = list(get_stop_words('en'))         #Have around 900 stopwords
nltk_words = list(stopwords.words('english'))   #Have around 150 stopwords
stop_words.extend(nltk_words)
stop_words = set(stop_words)
delete_stop_words = True


def is_interactive():
   return 'SHLVL' not in os.environ

if not is_interactive():
    def nop(it, *a, **k):
        return it


    tqdm = nop


# In[30]:



# record on tensorboard
save_dir_tb = './tensorboard'
if not os.path.exists(save_dir_tb):
    os.makedirs(save_dir_tb)
tbx = SummaryWriter(save_dir_tb)




def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything()




CRAWL_EMBEDDING_PATH = '../datasets/crawl-300d-2M.vec'
GLOVE_EMBEDDING_PATH = '../datasets/glove.840B.300d.txt'
NUM_MODELS = 2
LSTM_UNITS = 128
DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS
MAX_LEN = 220




# convert vector to an array
def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

# store word and vector in a dictionary
def load_embeddings(path):
    with open(path, encoding='utf8') as f:
        return dict(get_coefs(*line.strip().split(' ')) for line in tqdm(f))

# embedding_matrix[a word's index] is a row of vector represent that word
# for word not existent in database, it is a vector of zeros
def build_matrix(word_index, path):
    embedding_index = load_embeddings(path)
    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    unknown_words = []

    for word, i in word_index.items():
        try:
            embedding_matrix[i] = embedding_index[word]
        except KeyError:
            unknown_words.append(word)
    return embedding_matrix, unknown_words



# In[32]:



def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(y_pred, y_true):
    pred = torch.sigmoid(y_pred)

    pred[pred > 0.5] = 1
    pred[pred <= 0.5] = 0
    acc_general = torch.mean((pred == y_true).float()).item()
    acc_toxic = torch.mean((pred[y_true == 1] == 1).float()).item()
    acc_nontoxic = torch.mean((pred[y_true == 0] == 0).float()).item()
    return acc_general, acc_toxic, acc_nontoxic


def train_model(model, train, val, test, loss_fn, output_dim, lr=0.001,
                batch_size=512, n_epochs=4,
                enable_checkpoint_ensemble=True,
                validation_frequency=30):
    param_lrs = [{'params': param, 'lr': lr} for param in model.parameters()]
    optimizer = torch.optim.Adam(param_lrs, lr=lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.6 ** epoch)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle=False)
    all_test_preds = []
    checkpoint_weights = [2 ** epoch for epoch in range(n_epochs)]
    step = 0


    for epoch in range(n_epochs):
        start_time = time.time()

        scheduler.step()

        epoch_loss = 0.

        batches = 0

        for data in tqdm(train_loader, disable=False):
            # train
            model.train()
            x_batch = data[0]
            y_batch = data[1]
            x_batch = x_batch.to("cuda")
            y_batch = y_batch.to("cuda")

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, y_batch)

            weights = torch.zeros(y_batch.size()).cuda()

            weights[y_batch[:, 0] > 0.5] = 0.9
            weights[y_batch[:, 0] < 0.5] = 0.1
            loss_weighted = (loss * weights).mean()

            optimizer.zero_grad()
            loss_weighted.backward()

            optimizer.step()
            epoch_loss += loss_weighted.item() / len(train_loader)
            batch_loss = loss_weighted.item()
            with torch.no_grad():

                acc, acc_toxic, acc_nontoxic = accuracy(y_pred[:, 0], y_batch[:, 0])
            tbx.add_scalar('train/loss', batch_loss, step)
            tbx.add_scalar('train/acc', acc, step)
            tbx.add_scalar('train/acc_toxic', acc_toxic, step)
            tbx.add_scalar('train/acc_nontoxic', acc_nontoxic, step)


            batches += 1
            step += batch_size
            if batches % validation_frequency == 0:
                # validation
                model.eval()
                with torch.no_grad():
                    val_acc = 0

                    val_acc_toxic = 0
                    val_acc_nontoxic = 0
                    val_loss = 0
                    for x_y in val_loader:
                        x_batch = x_y[0]
                        y_batch = x_y[1]
                        x_batch = x_batch.cuda()
                        y_batch = y_batch.cuda()
                        y_pred = model(x_batch)
                        loss = loss_fn(y_pred, y_batch)
                        weights = torch.zeros(y_batch.size()).cuda()

                        weights[y_batch[:, 0] > 0.5] = 0.9
                        weights[y_batch[:, 0] < 0.5] = 0.1
                        loss_weighted = (loss * weights).mean()
                        val_loss += loss_weighted.item() / len(val_loader)

                        val_acc += accuracy(y_pred[:, 0], y_batch[:, 0])[0] / len(val_loader)
                        val_acc_toxic += accuracy(y_pred[:, 0], y_batch[:, 0])[1] / len(val_loader)
                        val_acc_nontoxic += accuracy(y_pred[:, 0], y_batch[:, 0])[2] / len(val_loader)
                    tbx.add_scalar('val/loss', val_loss, step)
                    tbx.add_scalar('val/acc', val_acc, step)
                    tbx.add_scalar('val/acc_toxic', val_acc_toxic, step)
                    tbx.add_scalar('val/acc_nontoxic', val_acc_nontoxic, step)

        # test
        model.eval()
        test_preds = np.zeros((len(test), output_dim))

        for i, x_batch in enumerate(test_loader):
            x_batch = x_batch[0].to("cuda")
            y_pred = sigmoid(model(x_batch).detach().cpu().numpy())

            test_preds[i * batch_size:(i+1) * batch_size, :] = y_pred

        all_test_preds.append(test_preds)
        elapsed_time = time.time() - start_time
        print('Epoch {}/{} \t loss={:.4f} \t time={:.2f}s'.format(
              epoch + 1, n_epochs, epoch_loss, elapsed_time))

    if enable_checkpoint_ensemble:

        test_preds = np.average(all_test_preds, weights=checkpoint_weights, axis=0)
    else:
        test_preds = all_test_preds[-1]

    return test_preds


# In[10]:



class SpatialDropout(nn.Dropout2d):
    def forward(self, x):
        x = x.unsqueeze(2)    # (N, T, 1, K)
        x = x.permute(0, 3, 2, 1)  # (N, K, 1, T)
        x = super(SpatialDropout, self).forward(x)  # (N, K, 1, T), some features are masked
        x = x.permute(0, 3, 2, 1)  # (N, T, 1, K)
        x = x.squeeze(2)  # (N, T, K)
        return x

class NeuralNet(nn.Module):
    def __init__(self, embedding_matrix, num_aux_targets):
        super(NeuralNet, self).__init__()
        embed_size = embedding_matrix.shape[1]



        self.embedding = nn.Embedding(max_features, embed_size)
        self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))
        self.embedding.weight.requires_grad = False
        self.embedding_dropout = SpatialDropout(0.3)


        self.lstm1 = nn.LSTM(embed_size, LSTM_UNITS, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(LSTM_UNITS * 2, LSTM_UNITS, bidirectional=True, batch_first=True)

        self.linear1 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)
        self.linear2 = nn.Linear(DENSE_HIDDEN_UNITS, DENSE_HIDDEN_UNITS)

        self.linear_out = nn.Linear(DENSE_HIDDEN_UNITS, 1)
        self.linear_aux_out = nn.Linear(DENSE_HIDDEN_UNITS, num_aux_targets)

    def forward(self, x):
        h_embedding = self.embedding(x)
        h_embedding = self.embedding_dropout(h_embedding)

        h_lstm1, _ = self.lstm1(h_embedding)
        h_lstm2, _ = self.lstm2(h_lstm1)


        # global average pooling
        avg_pool = torch.mean(h_lstm2, 1)
        # global max pooling
        max_pool, _ = torch.max(h_lstm2, 1)


        h_conc = torch.cat((max_pool, avg_pool), 1)
        h_conc_linear1  = F.relu(self.linear1(h_conc))
        h_conc_linear2  = F.relu(self.linear2(h_conc))

        hidden = h_conc + h_conc_linear1 + h_conc_linear2

        result = self.linear_out(hidden)
        aux_result = self.linear_aux_out(hidden)
        out = torch.cat([result, aux_result], 1)

        return out


# In[4]:


# def preprocess(data):
#     '''
#     Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
#     '''
#     punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
#     def clean_special_chars(text, punct):
#         for p in punct:
#             text = text.replace(p, ' ')
#         return text

#     data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
#     return data

def preprocess(data,delete_stop_words=False):

    '''
    Credit goes to https://www.kaggle.com/gpreda/jigsaw-fast-compact-solution
    '''
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~`" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
    def clean_special_chars(text, punct):
        for p in punct:
            text = text.replace(p, ' ')
        return text


    def remove_stop_words(x):
        output = ""
        tokens = text.text_to_word_sequence(x)
        for words in tokens:
            if not words in stop_words:
                output += words + " "
        return output


    data = data.astype(str).apply(lambda x: clean_special_chars(x, punct))
    if delete_stop_words:
        data = data.astype(str).apply(lambda x: remove_stop_words(x))
    return data


# In[ ]:


# print(x_train_torch)


# In[ ]:


contraction_mapping = {
    "Trump's" : 'trump is',"'cause": 'because',',cause': 'because',';cause': 'because',"ain't": 'am not','ain,t': 'am not',
    'ain;t': 'am not','ain´t': 'am not','ain’t': 'am not',"aren't": 'are not',
    'aren,t': 'are not','aren;t': 'are not','aren´t': 'are not','aren’t': 'are not',"can't": 'cannot',"can't've": 'cannot have','can,t': 'cannot','can,t,ve': 'cannot have',
    'can;t': 'cannot','can;t;ve': 'cannot have',
    'can´t': 'cannot','can´t´ve': 'cannot have','can’t': 'cannot','can’t’ve': 'cannot have',
    "could've": 'could have','could,ve': 'could have','could;ve': 'could have',"couldn't": 'could not',"couldn't've": 'could not have','couldn,t': 'could not','couldn,t,ve': 'could not have','couldn;t': 'could not',
    'couldn;t;ve': 'could not have','couldn´t': 'could not',
    'couldn´t´ve': 'could not have','couldn’t': 'could not','couldn’t’ve': 'could not have','could´ve': 'could have',
    'could’ve': 'could have',"didn't": 'did not','didn,t': 'did not','didn;t': 'did not','didn´t': 'did not',
    'didn’t': 'did not',"doesn't": 'does not','doesn,t': 'does not','doesn;t': 'does not','doesn´t': 'does not',
    'doesn’t': 'does not',"don't": 'do not','don,t': 'do not','don;t': 'do not','don´t': 'do not','don’t': 'do not',
    "hadn't": 'had not',"hadn't've": 'had not have','hadn,t': 'had not','hadn,t,ve': 'had not have','hadn;t': 'had not',
    'hadn;t;ve': 'had not have','hadn´t': 'had not','hadn´t´ve': 'had not have','hadn’t': 'had not','hadn’t’ve': 'had not have',"hasn't": 'has not','hasn,t': 'has not','hasn;t': 'has not','hasn´t': 'has not','hasn’t': 'has not',
    "haven't": 'have not','haven,t': 'have not','haven;t': 'have not','haven´t': 'have not','haven’t': 'have not',"he'd": 'he would',
    "he'd've": 'he would have',"he'll": 'he will',
    "he's": 'he is','he,d': 'he would','he,d,ve': 'he would have','he,ll': 'he will','he,s': 'he is','he;d': 'he would',
    'he;d;ve': 'he would have','he;ll': 'he will','he;s': 'he is','he´d': 'he would','he´d´ve': 'he would have','he´ll': 'he will',
    'he´s': 'he is','he’d': 'he would','he’d’ve': 'he would have','he’ll': 'he will','he’s': 'he is',"how'd": 'how did',"how'll": 'how will',
    "how's": 'how is','how,d': 'how did','how,ll': 'how will','how,s': 'how is','how;d': 'how did','how;ll': 'how will',
    'how;s': 'how is','how´d': 'how did','how´ll': 'how will','how´s': 'how is','how’d': 'how did','how’ll': 'how will',
    'how’s': 'how is',"i'd": 'i would',"i'll": 'i will',"i'm": 'i am',"i've": 'i have','i,d': 'i would','i,ll': 'i will',
    'i,m': 'i am','i,ve': 'i have','i;d': 'i would','i;ll': 'i will','i;m': 'i am','i;ve': 'i have',"isn't": 'is not',
    'isn,t': 'is not','isn;t': 'is not','isn´t': 'is not','isn’t': 'is not',"it'd": 'it would',"it'll": 'it will',"It's":'it is',
    "it's": 'it is','it,d': 'it would','it,ll': 'it will','it,s': 'it is','it;d': 'it would','it;ll': 'it will','it;s': 'it is','it´d': 'it would','it´ll': 'it will','it´s': 'it is',
    'it’d': 'it would','it’ll': 'it will','it’s': 'it is',
    'i´d': 'i would','i´ll': 'i will','i´m': 'i am','i´ve': 'i have','i’d': 'i would','i’ll': 'i will','i’m': 'i am',
    'i’ve': 'i have',"let's": 'let us','let,s': 'let us','let;s': 'let us','let´s': 'let us',
    'let’s': 'let us',"ma'am": 'madam','ma,am': 'madam','ma;am': 'madam',"mayn't": 'may not','mayn,t': 'may not','mayn;t': 'may not',
    'mayn´t': 'may not','mayn’t': 'may not','ma´am': 'madam','ma’am': 'madam',"might've": 'might have','might,ve': 'might have','might;ve': 'might have',"mightn't": 'might not','mightn,t': 'might not','mightn;t': 'might not','mightn´t': 'might not',
    'mightn’t': 'might not','might´ve': 'might have','might’ve': 'might have',"must've": 'must have','must,ve': 'must have','must;ve': 'must have',
    "mustn't": 'must not','mustn,t': 'must not','mustn;t': 'must not','mustn´t': 'must not','mustn’t': 'must not','must´ve': 'must have',
    'must’ve': 'must have',"needn't": 'need not','needn,t': 'need not','needn;t': 'need not','needn´t': 'need not','needn’t': 'need not',"oughtn't": 'ought not','oughtn,t': 'ought not','oughtn;t': 'ought not',
    'oughtn´t': 'ought not','oughtn’t': 'ought not',"sha'n't": 'shall not','sha,n,t': 'shall not','sha;n;t': 'shall not',"shan't": 'shall not',
    'shan,t': 'shall not','shan;t': 'shall not','shan´t': 'shall not','shan’t': 'shall not','sha´n´t': 'shall not','sha’n’t': 'shall not',
    "she'd": 'she would',"she'll": 'she will',"she's": 'she is','she,d': 'she would','she,ll': 'she will',
    'she,s': 'she is','she;d': 'she would','she;ll': 'she will','she;s': 'she is','she´d': 'she would','she´ll': 'she will',
    'she´s': 'she is','she’d': 'she would','she’ll': 'she will','she’s': 'she is',"should've": 'should have','should,ve': 'should have','should;ve': 'should have',
    "shouldn't": 'should not','shouldn,t': 'should not','shouldn;t': 'should not','shouldn´t': 'should not','shouldn’t': 'should not','should´ve': 'should have',
    'should’ve': 'should have',"that'd": 'that would',"that's": 'that is','that,d': 'that would','that,s': 'that is','that;d': 'that would',
    'that;s': 'that is','that´d': 'that would','that´s': 'that is','that’d': 'that would','that’s': 'that is',"there'd": 'there had',
    "there's": 'there is','there,d': 'there had','there,s': 'there is','there;d': 'there had','there;s': 'there is',
    'there´d': 'there had','there´s': 'there is','there’d': 'there had','there’s': 'there is',
    "they'd": 'they would',"they'll": 'they will',"they're": 'they are',"they've": 'they have',
    'they,d': 'they would','they,ll': 'they will','they,re': 'they are','they,ve': 'they have','they;d': 'they would','they;ll': 'they will','they;re': 'they are',
    'they;ve': 'they have','they´d': 'they would','they´ll': 'they will','they´re': 'they are','they´ve': 'they have','they’d': 'they would','they’ll': 'they will',
    'they’re': 'they are','they’ve': 'they have',"wasn't": 'was not','wasn,t': 'was not','wasn;t': 'was not','wasn´t': 'was not',
    'wasn’t': 'was not',"we'd": 'we would',"we'll": 'we will',"we're": 'we are',"we've": 'we have','we,d': 'we would','we,ll': 'we will',
    'we,re': 'we are','we,ve': 'we have','we;d': 'we would','we;ll': 'we will','we;re': 'we are','we;ve': 'we have',
    "weren't": 'were not','weren,t': 'were not','weren;t': 'were not','weren´t': 'were not','weren’t': 'were not','we´d': 'we would','we´ll': 'we will',
    'we´re': 'we are','we´ve': 'we have','we’d': 'we would','we’ll': 'we will','we’re': 'we are','we’ve': 'we have',"what'll": 'what will',"what're": 'what are',"what's": 'what is',
    "what've": 'what have','what,ll': 'what will','what,re': 'what are','what,s': 'what is','what,ve': 'what have','what;ll': 'what will','what;re': 'what are',
    'what;s': 'what is','what;ve': 'what have','what´ll': 'what will',
    'what´re': 'what are','what´s': 'what is','what´ve': 'what have','what’ll': 'what will','what’re': 'what are','what’s': 'what is',
    'what’ve': 'what have',"where'd": 'where did',"where's": 'where is','where,d': 'where did','where,s': 'where is','where;d': 'where did',
    'where;s': 'where is','where´d': 'where did','where´s': 'where is','where’d': 'where did','where’s': 'where is',
    "who'll": 'who will',"who's": 'who is','who,ll': 'who will','who,s': 'who is','who;ll': 'who will','who;s': 'who is',
    'who´ll': 'who will','who´s': 'who is','who’ll': 'who will','who’s': 'who is',"won't": 'will not','won,t': 'will not','won;t': 'will not',
    'won´t': 'will not','won’t': 'will not',"wouldn't": 'would not','wouldn,t': 'would not','wouldn;t': 'would not','wouldn´t': 'would not',
    'wouldn’t': 'would not',"you'd": 'you would',"you'll": 'you will',"you're": 'you are','you,d': 'you would','you,ll': 'you will',
    'you,re': 'you are','you;d': 'you would','you;ll': 'you will',
    'you;re': 'you are','you´d': 'you would','you´ll': 'you will','you´re': 'you are','you’d': 'you would','you’ll': 'you will','you’re': 'you are',
    '´cause': 'because','’cause': 'because',"you've": "you have","could'nt": 'could not',
    "havn't": 'have not',"here’s": "here is",'i""m': 'i am',"i'am": 'i am',"i'l": "i will","i'v": 'i have',"wan't": 'want',"was'nt": "was not","who'd": "who would",
    "who're": "who are","who've": "who have","why'd": "why would","would've": "would have","y'all": "you all","y'know": "you know","you.i": "you i",
    "your'e": "you are","arn't": "are not","agains't": "against","c'mon": "common","doens't": "does not",'don""t': "do not","dosen't": "does not",
    "dosn't": "does not","shoudn't": "should not","that'll": "that will","there'll": "there will","there're": "there are",
    "this'll": "this all","u're": "you are", "ya'll": "you all","you'r": "you are","you’ve": "you have","d'int": "did not","did'nt": "did not","din't": "did not","dont't": "do not","gov't": "government",
    "i'ma": "i am","is'nt": "is not","‘I":'I',
    'ᴀɴᴅ':'and','ᴛʜᴇ':'the','ʜᴏᴍᴇ':'home','ᴜᴘ':'up','ʙʏ':'by','ᴀᴛ':'at','…and':'and','civilbeat':'civil beat',\
    'TrumpCare':'Trump care','Trumpcare':'Trump care', 'OBAMAcare':'Obama care','ᴄʜᴇᴄᴋ':'check','ғᴏʀ':'for','ᴛʜɪs':'this','ᴄᴏᴍᴘᴜᴛᴇʀ':'computer',\
    'ᴍᴏɴᴛʜ':'month','ᴡᴏʀᴋɪɴɢ':'working','ᴊᴏʙ':'job','ғʀᴏᴍ':'from','Sᴛᴀʀᴛ':'start','gubmit':'submit','CO₂':'carbon dioxide','ғɪʀsᴛ':'first',\
    'ᴇɴᴅ':'end','ᴄᴀɴ':'can','ʜᴀᴠᴇ':'have','ᴛᴏ':'to','ʟɪɴᴋ':'link','ᴏғ':'of','ʜᴏᴜʀʟʏ':'hourly','ᴡᴇᴇᴋ':'week','ᴇɴᴅ':'end','ᴇxᴛʀᴀ':'extra',\
    'Gʀᴇᴀᴛ':'great','sᴛᴜᴅᴇɴᴛs':'student','sᴛᴀʏ':'stay','ᴍᴏᴍs':'mother','ᴏʀ':'or','ᴀɴʏᴏɴᴇ':'anyone','ɴᴇᴇᴅɪɴɢ':'needing','ᴀɴ':'an','ɪɴᴄᴏᴍᴇ':'income',\
    'ʀᴇʟɪᴀʙʟᴇ':'reliable','ғɪʀsᴛ':'first','ʏᴏᴜʀ':'your','sɪɢɴɪɴɢ':'signing','ʙᴏᴛᴛᴏᴍ':'bottom','ғᴏʟʟᴏᴡɪɴɢ':'following','Mᴀᴋᴇ':'make',\
    'ᴄᴏɴɴᴇᴄᴛɪᴏɴ':'connection','ɪɴᴛᴇʀɴᴇᴛ':'internet','financialpost':'financial post', 'ʜaᴠᴇ':' have ', 'ᴄaɴ':' can ', 'Maᴋᴇ':' make ', 'ʀᴇʟɪaʙʟᴇ':' reliable ', 'ɴᴇᴇᴅ':' need ',
    'ᴏɴʟʏ':' only ', 'ᴇxᴛʀa':' extra ', 'aɴ':' an ', 'aɴʏᴏɴᴇ':' anyone ', 'sᴛaʏ':' stay ', 'Sᴛaʀᴛ':' start', 'SHOPO':'shop',
    }
punct_mapping = {"‘": "'", "₹": "e", "´": "'", "°": "", "€": "e", "™": "tm", "√": " sqrt ", "×": "x", "²": "2", "—": "-", "–": "-", "’": "'", "_": "-", "`": "'", '“': '"', '”': '"', '“': '"', "£": "e", '∞': 'infinity', 'θ': 'theta', '÷': '/', 'α': 'alpha', '•': '.', 'à': 'a', '−': '-', 'β': 'beta', '∅': '', '³': '3', 'π': 'pi', }
punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’' + '∞θ÷α•à−β∅³π‘₹´°£€\×™√²—–&'
mispell_dict = {'SB91':'senate bill','tRump':'trump','utmterm':'utm term','FakeNews':'fake news','Gʀᴇat':'great','ʙᴏᴛtoᴍ':'bottom','washingtontimes':'washington times','garycrum':'gary crum','htmlutmterm':'html utm term','RangerMC':'car','TFWs':'tuition fee waiver','SJWs':'social justice warrior','Koncerned':'concerned','Vinis':'vinys','Yᴏᴜ':'you','Trumpsters':'trump','Trumpian':'trump','bigly':'big league','Trumpism':'trump','Yoyou':'you','Auwe':'wonder','Drumpf':'trump','utmterm':'utm term','Brexit':'british exit','utilitas':'utilities','ᴀ':'a', '😉':'wink','😂':'joy','😀':'stuck out tongue', 'theguardian':'the guardian','deplorables':'deplorable', 'theglobeandmail':'the globe and mail', 'justiciaries': 'justiciary','creditdation': 'Accreditation','doctrne':'doctrine','fentayal': 'fentanyl','designation-': 'designation','CONartist' : 'con-artist','Mutilitated' : 'Mutilated','Obumblers': 'bumblers','negotiatiations': 'negotiations','dood-': 'dood','irakis' : 'iraki','cooerate': 'cooperate','COx':'cox','racistcomments':'racist comments','envirnmetalists': 'environmentalists',}


# In[ ]:


def clean_preprocess(text, punct = punct, mapping = punct_mapping, mispell = mispell_dict):
    for p in mapping:
        text = text.replace(p, mapping[p])
    for p in punct:
        text = text.replace(p, ' ')
    specials = {'\u200b': ' ', '…': ' ... ', '\ufeff': '', 'करना': '', 'है': ''}  # Other special characters that I have to deal with in last
    for s in specials:
        text = text.replace(s, specials[s])
    for word in mispell.keys():
        text = text.replace(word, mispell[word])
    return text

# df['treated_comment'] = df['treated_comment'].apply(lambda x: clean_special_chars(x, punct, punct_mapping))
# vocab = build_vocab(df['treated_comment'])


# # Preprocessing

# In[3]:


train_val = pd.read_csv('../datasets/jigsaw-unintended-bias-in-toxicity-classification/train.csv')
test = pd.read_csv('../datasets/jigsaw-unintended-bias-in-toxicity-classification/test.csv')
random.seed(123)
indexes = random.sample(range(len(train_val)), k=len(train_val)//10)
val = train_val.loc[indexes, :]
train = train_val.drop(indexes, axis=0)

# print("train size: ", len(train))
# print("validation size: ", len(val))
# print("test size: ", len(test))
cleanpre = args.clean_pre

if (cleanpre):
    x_train = clean_preprocess(train['comment_text'])
    x_val = clean_preprocess(val['comment_text'])
    x_test = clean_preprocess(test['comment_text'])
else:
    x_train = preprocess(train['comment_text'])
    x_val = preprocess(val['comment_text'])
    x_test = preprocess(test['comment_text'])
y_train = np.where(train['target'] >= 0.5, 1, 0)
y_aux_train = train[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
# x_val = preprocess(val['comment_text'])
y_val = np.where(val['target'] >= 0.5, 1, 0)
y_aux_val = val[['target', 'severe_toxicity', 'obscene', 'identity_attack', 'insult', 'threat']]
# x_test = preprocess(test['comment_text'])


# In[13]:



max_features = None



# In[14]:


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(list(x_train) + list(x_test) + list(x_val)) # create an integer for every word

x_train = tokenizer.texts_to_sequences(x_train) # convert every text to a sequence of integers
x_val = tokenizer.texts_to_sequences(x_val)
x_test = tokenizer.texts_to_sequences(x_test)
x_train = sequence.pad_sequences(x_train, maxlen=MAX_LEN) # pad every sequence to make them of the same length
x_val = sequence.pad_sequences(x_val, maxlen=MAX_LEN)
x_test = sequence.pad_sequences(x_test, maxlen=MAX_LEN)



# In[15]:


max_features = max_features or len(tokenizer.word_index) + 1
max_features


# In[16]:


crawl_matrix, unknown_words_crawl = build_matrix(tokenizer.word_index, CRAWL_EMBEDDING_PATH)
# print('n unknown words (crawl): ', len(unknown_words_crawl))


# In[17]:


glove_matrix, unknown_words_glove = build_matrix(tokenizer.word_index, GLOVE_EMBEDDING_PATH)
# print('n unknown words (glove): ', len(unknown_words_glove))


# In[18]:


embedding_matrix = np.concatenate([crawl_matrix, glove_matrix], axis=-1)
embedding_matrix.shape

del crawl_matrix
del glove_matrix
gc.collect()



# In[19]:


x_train_torch = torch.tensor(x_train, dtype=torch.long)
x_val_torch = torch.tensor(x_val, dtype=torch.long)
x_test_torch = torch.tensor(x_test, dtype=torch.long)
y_train_torch = torch.tensor(np.hstack([y_train[:, np.newaxis], y_aux_train]), dtype=torch.float32)
y_val_torch = torch.tensor(np.hstack([y_val[:, np.newaxis], y_aux_val]), dtype=torch.float32)

# Training
# In[20]:


train_dataset = data.TensorDataset(x_train_torch, y_train_torch)
val_dataset = data.TensorDataset(x_val_torch, y_val_torch)
test_dataset = data.TensorDataset(x_test_torch)



# In[ ]:


model = NeuralNet(embedding_matrix, y_aux_train.shape[-1]).cuda()

test_preds = train_model(model, train_dataset, val_dataset, test_dataset,n_epochs = args.epoch, lr = args.lr, output_dim=y_train_torch.shape[-1],
                         loss_fn=nn.BCEWithLogitsLoss(reduction='none'))


# In[ ]:



submission = pd.DataFrame.from_dict({
    'id': test['id'],
    'prediction': test_preds[:, 0]
})

submission.to_csv('submission.csv', index=False)


# Note that the solution is not validated in this kernel. So for tuning anything, you should build a validation framework using e. g. KFold CV. If you just check what works best by submitting, you are very likely to overfit to the public LB.

# # Ways to improve this kernel

# This kernel is just a simple baseline kernel, so there are many ways to improve it. Some ideas to get you started:
# - Add a contraction mapping. E. g. mapping "is'nt" to "is not" can help the network because "not" is explicitly mentioned. They were very popular in the recent quora competition, see for example [this kernel](https://www.kaggle.com/theoviel/improve-your-score-with-some-text-preprocessing).
# - Try to reduce the number of words that are not found in the embeddings. At the moment, around 170k words are not found. We can take some steps to decrease this amount, for example trying to find a vector for a processed (capitalized, stemmed, ...) version of the word when the vector for the regular word can not be found. See the [3rd place solution](https://www.kaggle.com/wowfattie/3rd-place) of the quora competition for an excellent implementation of this.
# - Try cyclic learning rate (CLR). I have found CLR to almost always improve my network recently compared to the default parameters for Adam. In this case, we are already using a learning rate scheduler, so this might not be the case. But it is still worth to try it out. See for example my [my other PyTorch kernel](https://www.kaggle.com/bminixhofer/deterministic-neural-networks-using-pytorch) for an implementation of CLR in PyTorch.
# - Use sequence bucketing to train faster and fit more networks into the two hours. The winning team of the quora competition successfully used sequence bucketing to drastically reduce the time it took to train RNNs. An excerpt from their [solution summary](https://www.kaggle.com/c/quora-insincere-questions-classification/discussion/80568#latest-487092):

#
# > We aimed at combining as many models as possible. To do this, we needed to improve runtime and the most important thing to achieve this was the following. We do not pad sequences to the same length based on the whole data, but just on a batch level. That means we conduct padding and truncation on the data generator level for each batch separately, so that length of the sentences in a batch can vary in size. Additionally, we further improved this by not truncating based on the length of the longest sequence in the batch, but based on the 95% percentile of lengths within the sequence. This improved runtime heavily and kept accuracy quite robust on single model level, and improved it by being able to average more models.
#
# - Try a (weighted) average of embeddings instead of concatenating them. A 600d vector for each word is a lot, it might work better to average them instead. See [this paper](https://www.aclweb.org/anthology/N18-2031) for why this even works.
# - Limit the maximum number of words used to train the NN. At the moment, there is no limit set to the maximum number of words in the tokenizer, so we use every word that occurs in the training data, even if it is only mentioned once. This could lead to overfitting so it might be better to limit the maximum number of words to e. g. 100k.
#

# Thanks for reading. Good luck and have fun in this competition!
