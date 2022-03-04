import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
from utils import pre_process


CLEAN_STOPWORDS = True


class tokenizer(object):
    def __init__(self, vocab_path=None):
        # 加载词表，如果有的话
        if vocab_path is not None:
            self.vocab = np.load(vocab_path, allow_pickle=True).item()
            self.vocab_size = len(self.vocab)
            print("字典大小：", self.vocab_size)

    # 将一个句子分词
    def split_text(self, sentence: str):
        words = pre_process(sentence).split()   # 数据预处理
        # 去除停用词
        if CLEAN_STOPWORDS:
            stop_words = stopwords.words('english')
            words = [word for word in words if word not in stop_words]
        return np.array(words)

    # 在一组句子上生成词表（字典），并存储在vocab.npy
    def generate_vocab(self, sentences, min_freq=2):
        vocab = {}
        for sentence in tqdm(sentences):
            words = self.split_text(sentence)
            for word in words:
                # 统计所有词的词频
                if word in vocab.keys():
                    vocab[word] += 1
                else:
                    vocab[word] = 1
        # 去除词频小于等于min_freq的词
        for word in list(vocab.keys()):
            if vocab[word] <= min_freq:
                vocab.pop(word)
        np.save("vocab.npy", vocab)
        print("词表大小：", len(vocab))
        self.vocab = vocab
        self.vocab_size = len(vocab)

    # 生成BOW文本表示
    def generate_bow(self, sentences):
        sents_bow = np.empty(shape=[0, self.vocab_size])
        for sentence in sentences:
            vocab_freq = {word: 0 for word in self.vocab.keys()}
            words = self.split_text(sentence)
            for word in words:
                vocab_freq[word] += 1
            sent_bow = np.array(list(vocab_freq.values()))
            sents_bow = np.concatenate((sents_bow, [sent_bow]), axis=0)
        return sents_bow
