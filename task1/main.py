import random

import numpy as np
from data_loader import data_loader, dataset
from vectorizer import vectorizer
from utils import cross_entropy, one_hot, eval
from model import SoftmaxClassifier


train_path = "./data/train.tsv"
test_path = "./data/test.tsv"
batch_size = 32


def train(model, train_loader, test_loader, vectorizer, epoch=10, lr=0.01, log_interval=500):
    # all_x = train_loader.dataset.x[:10000]
    # all_y = train_loader.dataset.y[:10000]
    # all_x = vectorizer.tri_vectorizer.transform(all_x).A  # (batch_size, vocab_size)
    # all_x = np.insert(all_x, 0, values=1, axis=1)  # 加上一行1用于与b相乘

    test_x = train_loader.dataset.x[:10000]
    test_y = test_loader.dataset.y[:10000]
    test_x = vectorizer.tri_vectorizer.transform(test_x).A  # (batch_size, vocab_size)
    test_x = np.insert(test_x, 0, values=1, axis=1)  # 加上一行1用于与b相乘

    for epoch_id in range(1, epoch+1):
        # y_pred = model.predict(all_x)
        # eval(y_pred, all_y)
        y_pred = model.predict(test_x)
        eval(y_pred, test_y)
        output = model.forward(test_x)
        loss = cross_entropy(output, one_hot(test_y))
        print("测试集loss: ", loss)

        for batch_idx, (x, target) in enumerate(train_loader):
            x = vectorizer.tri_vectorizer.transform(x).A  # (batch_size, vocab_size)
            x = np.insert(x, 0, values=1, axis=1)  # 加上一行1用于与b相乘
            target = one_hot(target)  # (batch_size, num_classes)

            # print(batch_idx)
            model.step(x, target, lr)

            # 输出loss
            if batch_idx % log_interval == 0:
                # test_nums = 5000
                # test_idx = random.randint(0, 100000)
                # x = train_loader.dataset.x[test_idx:test_idx+test_nums]
                # y = train_loader.dataset.y[test_idx:test_idx+test_nums]
                # x = vectorizer.tri_vectorizer.transform(x).A  # (batch_size, vocab_size)
                # x = np.insert(x, 0, values=1, axis=1)  # 加上一行1用于与b相乘
                # target = one_hot(y)  # (batch_size, num_classes)
                # output = model.forward(x)
                # loss = cross_entropy(output, target)

                output = model.forward(x)
                loss = cross_entropy(output, target)
                percent = 100. * batch_idx * batch_size / len(train_loader.dataset.x)
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch_id, batch_idx * batch_size, len(train_loader.dataset.x),
                    percent, loss.item()))


if __name__ == "__main__":
    train_loader = data_loader(batch_size)
    train_dataset = train_loader.load(train_path)

    test_loader = data_loader(batch_size)
    test_dataset = test_loader.load(test_path)

    # 加载vectorizer，并统计训练集数据
    print("加载vectorizer")
    vectorizer = vectorizer(train_loader.dataset.x)
    vocab_size = len(vectorizer.tri_vectorizer.vocabulary_)
    # TODO 存储和恢复vectorizer

    model = SoftmaxClassifier(vocab_size=vocab_size)
    train(model, train_loader=train_loader, test_loader=test_loader, vectorizer=vectorizer)

    # print("向量化")
    # x_train_vec = vectorizer.vectorize(train_dataset.x)
    # y_train = train_data_loader.y
    # x_test_vec = vectorizer.vectorize(test_data_loader.x)
    # y_test = test_data_loader.y
