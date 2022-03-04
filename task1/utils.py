import re
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.metrics import precision_score, accuracy_score


# 预处理文本：全部转小写、去除标点符号
def pre_process(text):
    text = text.lower()     # 转小写
    # 去除标点符号
    punctuation = '!,;:?"\'、，；'
    text = re.sub(r'[{}]+'.format(punctuation), ' ', text)
    return text.strip()


def softmax(x):
    x = x - np.max(x)
    exp_x = np.exp(x)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x


def one_hot(x, num_class=5):
    res = np.zeros((len(x), num_class))
    res[range(len(x)), x] = 1
    return res


# 交叉熵代价函数
def cross_entropy(output, target):
    # output: (batch_size, num_class)
    delta = 1e-7  # 添加一个微小值可以防止负无限大(np.log(0))的发生。
    # target = one_hot(target)    # (batch_size, num_class)
    return -np.sum(target * np.log(output + delta)) / output.shape[0]


def eval(y_pred, y_true):
    print("准确率：", accuracy_score(y_true, y_pred)) # 0.2222222222222222
    print("直接平均精确率：", precision_score(y_true, y_pred, average='macro'))  # 0.2222222222222222
    print("加权平均精确率：", precision_score(y_true, y_pred, average='weighted'))  # 0.2222222222222222


def sklearn_train(x_train_vec, y_train, x_test_vec, y_test):
    clf = SGDClassifier(alpha=0.001,
                        loss='log',  # hinge代表SVM，log是逻辑回归
                        early_stopping=False,
                        eta0=0.001,
                        learning_rate='adaptive',  # constant、optimal、invscaling、adaptive
                        max_iter=100
                        )
    # 打乱数据，训练
    x_train_vec, y_train = shuffle(x_train_vec, y_train)
    x_test_vec, y_test = shuffle(x_test_vec, y_test)

    # print("开始训练模型")
    # clf.partial_fit(x_train_vec, y_train, classes=np.unique(train_data_loader.y))
    # # clf.fit(x_train_vec, y_train)
    #
    # # 预测训练集
    # print("开始评估训练集")
    # predict = clf.predict(x_train_vec)
    # # 训练集的评估
    # print(np.mean(predict == y_train))
    #
    # print("开始评估测试集")
    # test_predict = clf.predict(x_test_vec)
    # # 训练集的评估
    # print(np.mean(test_predict == y_test))

