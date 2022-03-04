from data_loader import dataset, data_loader
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


class vectorizer(object):
    # X为训练集所有的句子数据，统计得到词表
    def __init__(self, X):
        # self.tri_vectorizer = TfidfVectorizer(ngram_range=(2, 3))
        self.tri_vectorizer = CountVectorizer(ngram_range=(2, 3))
        self.tri_vectorizer.fit(X)

    # 将句子转成句向量
    def vectorize(self, data):
        return self.tri_vectorizer.transform(data).toarray()


train_path = "./data/train.tsv"
test_path = "./data/test.tsv"

# 3-gram：76834
# 2-gram：58010
# 1-gram：12680

if __name__ == "__main__":
    train_data_loader = data_loader()
    train_dataset = train_data_loader.load(train_path)
    vectorizer = vectorizer(train_data_loader.x)
    print(len(vectorizer.tri_vectorizer.vocabulary_))