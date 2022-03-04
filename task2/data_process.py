import re
import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtext.vocab import vocab
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# nltk.download('punkt')

MAX_LEN = 25  # 将每条评论通过截断或者补0，使得长度变成500


def save_img(loss, acc, test_acc):
    num_epochs = len(loss)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, test_acc, 'r', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.savefig("acc1.png")

    plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("loss1.png")


def load_pretrained_embedding(words, pretrained_vocab):
    '''从预训练好的vocab中提取出words对应的词向量'''
    embed = torch.zeros(len(words), pretrained_vocab.vectors[0].shape[0]) # 初始化为0
    oov_count = 0 # out of vocabulary
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocab.stoi[word]
            embed[i, :] = pretrained_vocab.vectors[idx]
        except KeyError:
            oov_count += 1
    if oov_count > 0:
        print("There are %d oov words." % oov_count)
    return embed


def load_data(data_path):
    data = pd.read_csv(data_path, sep="\t")
    print("data.shape: ", data.shape)  # (156060, 4)
    # 提取句子与标签的列
    x = data["Phrase"]
    y = data["Sentiment"]
    return x, torch.tensor(y)


# 预处理文本：全部转小写、去除标点符号
def pre_process(text):
    text = text.lower()  # 转小写
    # 去除标点符号
    punctuation = '!,;:?."\'、，；`'
    text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
    return text.strip()


# 分词
def get_tokenized_sent(sents):
    stopWords = set(stopwords.words('english'))
    def tokenizer(text):
        words = word_tokenize(text)
        words = [word.lower() for word in words
                 if word.isalpha() and word not in stopWords]
        return words
    return [tokenizer(review) for review in sents]


# 获得数据集的词典
def get_vocab(sents):
    tokenized_data = get_tokenized_sent(sents)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    min_feq = 3
    idx = len(sorted_by_freq_tuples) - 1
    while sorted_by_freq_tuples[idx][1] < min_feq:
        sorted_by_freq_tuples.pop(idx)
        idx -= 1
    # 用<NOF>表示未找到的词
    sorted_by_freq_tuples.append(('<NOF>', 1))
    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab_obj = vocab(ordered_dict)
    return vocab_obj


def words2id(vocab_dic, words):
    def pad(x):
        return x[:MAX_LEN] if len(x) > MAX_LEN else x + [0] * (MAX_LEN - len(x))

    vec = []
    not_found_id = len(vocab_dic) - 1
    for word in words:
        try:
            vec.append(vocab_dic[word])
        except KeyError:
            vec.append(not_found_id)
    return pad(vec)


# 将句子转成长度一致的 词序号向量
def preprocess_data(sents, vocab_dic, file_name):
    tokenized_data = get_tokenized_sent(sents)
    list = []
    for words in tqdm(tokenized_data):
        list.append(words2id(vocab_dic, words))
    # np.save(file_name, np.array(list))
    return torch.tensor(np.array(list))


def get_sents_ids(file_path):
    sents_ids = np.load(file_path).tolist()
    return torch.tensor(sents_ids)


def analysis_len(sents):
    sents_len = [len(sent) for sent in sents]
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.figure(figsize=(30, 12), dpi=100)
    plt.subplot(2, 3, 2)
    plt.title("句子长度分布")
    plt.hist(sents_len, bins=list(range(0, max(sents_len) + 1, 1)))
    plt.xlabel('句子长度')
    plt.ylabel('句子数量')
    """ title 累计分布"""
    plt.subplot(2, 3, 5)
    plt.title('累计分布图')
    plt.hist(sents_len, bins=list(range(0, max(sents_len) + 1, 1)), cumulative=True)
    plt.xlabel('句子长度')
    plt.ylabel('累计比例(%)')

    plt.savefig("sent_len.png")