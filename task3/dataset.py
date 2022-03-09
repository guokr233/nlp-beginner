import jsonlines as jl
from torch.utils.data import Dataset, DataLoader
import pickle
import collections
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchtext.vocab import vocab
from collections import OrderedDict
from tqdm import tqdm
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


pad_id = 0
not_found_id = 1


class SentPairDataset(Dataset):
    def __init__(self, data_path, max_len, data_mode):
        '''
        :param data_path: 数据集路径
        :param w2id_dict: word2id的字典
        :param max_len: 要求截断或填充至的句子长度
        :param data_mode: train or test，用于命名文件
        '''
        self.labels = load_data_from_jsonl(data_path, only_labels=True)
        sents1_and_len = load_sents_and_len("./data/" + data_mode + "_sent1")
        sents2_and_len = load_sents_and_len("./data/" + data_mode + "_sent2")
        self.sent1_list = np.array(sents1_and_len["sents"])
        self.sent2_list = np.array(sents2_and_len["sents"])
        self.sent1_lens = np.array(sents1_and_len["sents_length"])
        self.sent2_lens = np.array(sents2_and_len["sents_length"])
        self.sent1_list = torch.from_numpy(self.sent1_list)
        self.sent2_list = torch.from_numpy(self.sent2_list)
        self.sent1_lens = torch.from_numpy(self.sent1_lens)
        self.sent2_lens = torch.from_numpy(self.sent2_lens)
        # self.sent1_list = torch.from_numpy(self.sent1_list).type(torch.LongTensor)
        # self.sent2_list = torch.from_numpy(self.sent2_list).type(torch.LongTensor)
        # self.sent1_lens = torch.from_numpy(self.sent1_lens).type(torch.LongTensor)
        # self.sent2_lens = torch.from_numpy(self.sent2_lens).type(torch.LongTensor)
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sent1_list[idx], self.sent1_lens[idx], self.sent2_list[idx], \
               self.sent2_lens[idx], self.labels[idx]


def load_data_from_jsonl(file_name, only_labels=False):
    labels2id = {"entailment": 0, "neutral": 1, "contradiction": 2}
    labels = []
    if only_labels:
        with open(file_name, "r+") as f:
            for row in jl.Reader(f):
                if row["gold_label"] != "-":
                    labels.append(labels2id[row["gold_label"]])
        return labels
    sentence1_list = []
    sentence2_list = []
    with open(file_name, "r+") as f:
        for row in jl.Reader(f):
            if row["gold_label"] != "-":
                labels.append(labels2id[row["gold_label"]])
                sentence1_list.append(row["sentence1"])
                sentence2_list.append(row["sentence2"])
    return sentence1_list, sentence2_list, labels


def load_sents_and_len(data_path):
    # Restore from a file
    f = open(data_path, 'rb')
    sents_and_len = pickle.load(f)
    return sents_and_len


# 分词
def get_tokenized_sent(sents):
    stopWords = set(stopwords.words('english'))
    def tokenizer(text):
        words = word_tokenize(text)
        words = [word.lower() for word in words
                 if word.isalpha() and word not in stopWords]
        if len(words) == 0:
            words = ["a"]   # 小埋一个坑
        return words
    return [tokenizer(review) for review in sents]


# 获得数据集的词典
def get_vocab(sents, min_feq=3, if_save=False, save_path="vocab"):
    tokenized_data = get_tokenized_sent(sents)
    counter = collections.Counter([tk for st in tokenized_data for tk in st])
    sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    idx = len(sorted_by_freq_tuples) - 1
    while sorted_by_freq_tuples[idx][1] < min_feq:
        sorted_by_freq_tuples.pop(idx)
        idx -= 1
    # 用<NOF>表示未找到的词
    sorted_by_freq_tuples.insert(0, ('<NOF>', 1))
    # 用<PAD>表示被填充的位置，序号为0
    sorted_by_freq_tuples.insert(0, ("<PAD>", 1))

    ordered_dict = OrderedDict(sorted_by_freq_tuples)
    vocab_obj = vocab(ordered_dict)
    print("字典大小： ", len(vocab_obj))
    if if_save:
        f = open(save_path, 'wb')
        pickle.dump(vocab_obj, f)
        print("----------------------字典存储完毕----------------------")
    return vocab_obj


def load_vocab(vocab_path):
    # Restore from a file
    f = open(vocab_path, 'rb')
    vocab_obj = pickle.load(f)
    return vocab_obj


def words2id(vocab_dic, words, max_len):
    def pad(x):
        return x[:max_len] if len(x) > max_len else x + [pad_id] * (max_len - len(x))

    vec = []
    not_found_id = 1
    for word in words:
        try:
            vec.append(vocab_dic[word])
        except KeyError:
            vec.append(not_found_id)
    return pad(vec)


def analysis_len(sents):
    sents_len = [len(sent) for sent in sents]
    sents_len = sorted(sents_len)
    nums = len(sents_len)
    print("70%: ", sents_len[int(0.7 * nums)])
    print("80%: ", sents_len[int(0.8 * nums)])
    print("90%: ", sents_len[int(0.9 * nums)])
    print("95%: ", sents_len[int(0.95 * nums)])
    print("99%: ", sents_len[int(0.99 * nums)])
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


def sents2matrix(sents, w2id_dict, max_len, if_save=True, save_path=None):
    tokenized_data = get_tokenized_sent(sents)
    sents_length = [min(len(sent), max_len) for sent in tokenized_data]
    matrix = []
    for words in tqdm(tokenized_data):
        matrix.append(words2id(w2id_dict, words, max_len))
    sents_and_len = {"sents": matrix, "sents_length": sents_length}
    if if_save:
        f = open(save_path, 'wb')
        pickle.dump(sents_and_len, f)
        print("----------------------词id矩阵及句长" + save_path + " 存储完毕----------------------")
    return sents_and_len


if __name__ == "__main__":
    train_path = "./data/snli_1.0_train.jsonl"
    test_path = "./data/snli_1.0_dev.jsonl"
    max_len = 25
    train_sentence1_list, train_sentence2_list, labels = load_data_from_jsonl(train_path)
    test_sentence1_list, test_sentence2_list, labels = load_data_from_jsonl(test_path)
    sents = train_sentence1_list + train_sentence2_list
    print("读取数据完毕")
    # analysis_len([sent.split() for sent in sents])  # 分析句子长度的分布
    vocab = get_vocab(sents, if_save=True, save_path="./data/vocab")  # 创建并存储字典
    w2id_dic = vocab.get_stoi()
    test_sents1 = sents2matrix(test_sentence1_list, w2id_dic, max_len, if_save=True, save_path="./data/test_sent1")
    test_sents2 = sents2matrix(test_sentence2_list, w2id_dic, max_len, if_save=True, save_path="./data/test_sent2")
    train_sents1 = sents2matrix(train_sentence1_list, w2id_dic, max_len, if_save=True, save_path="./data/train_sent1")
    train_sents2 = sents2matrix(train_sentence2_list, w2id_dic, max_len, if_save=True, save_path="./data/train_sent2")
