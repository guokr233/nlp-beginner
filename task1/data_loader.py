import numpy as np
import pandas as pd
from random import shuffle


class dataset:
    def __init__(self, x, y, shuffled=True, batch_size=32):
        # 打乱数据集
        if shuffled:
            zip_xy = list(zip(x, y))
            shuffle(zip_xy)
            origin = list(zip(*zip_xy))
            self.x = origin[0][:-1]
            self.y = origin[1][:-1]
        else:
            self.x = x
            self.y = y


class data_loader:
    def __init__(self, batch_size):
        self.labels = [1, 2, 3, 4]
        self.amount = None   # 数据的条数
        self.dataset = None
        self.batch_size = None
        self.now = 0  # 输出数据的位置
        self.batch_size = batch_size

    def load(self, data_dir, shuffled=True, batch_size=32):
        print("Start loading data...")
        data = pd.read_csv(data_dir, sep="\t")
        print("Load data over...")
        print("data.shape: ", data.shape)  # (156060, 4)
        print("data.keys: ", data.keys())  # ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']
        # 提取句子与标签的列
        x = data["Phrase"]
        y = data["Sentiment"]
        print("x.shape：", x.shape)
        self.amount = len(x)  # 数据的条数
        self.dataset = dataset(x, y, shuffled=shuffled)
        self.batch_size = batch_size

    def __iter__(self):
        return self

    def __next__(self):
        if self.now == -1:
            self.now = 0
            raise StopIteration()
        return self.get_batch()

    # 返回一个batch
    def get_batch(self):
        new_now = self.now + self.batch_size
        if new_now > self.amount:   # 避免越界
            new_now = -1
        x = np.array(self.dataset.x[self.now: new_now])
        y = np.array(self.dataset.y[self.now: new_now])
        self.now = new_now
        return x, y


all_sentences = ["Joe Joe waited for the train",
                 "The train was late",
                 "Mary and Samantha took the bus",
                 "I looked for Mary and Samantha at the bus station",
                 "Mary and Samantha arrived at the bus station early but waited until noon for the bus"]
sentence_arr = np.array(all_sentences)
