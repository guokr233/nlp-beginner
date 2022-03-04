import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import time
from data_process import preprocess_data, load_data, \
    get_vocab, get_sents_ids, load_pretrained_embedding, save_img
from RNNClassifier import RNNClassifier
import torchtext.vocab as Vocab
import os
import pandas as pd
from sklearn.model_selection import train_test_split


""" 设置随机种子 """
torch.manual_seed(42)
np.random.seed(42)

train_path = "./data/train2.tsv"
test_path = "./data/test2.tsv"

embed_size = 100
num_hiddens = 100
num_layers = 1
lr = 0.01
num_epochs = 5
batch_size = 32

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_ROOT = "./data"
glove_vocab = Vocab.GloVe(name='6B', dim=embed_size, cache=os.path.join(DATA_ROOT, "glove"))


# def train2(model, optimizer, train_loader:DataLoader, batch_size, epochs):
#     for epoch in range(1, epochs):
#         process_bar = tqdm(train_loader)  # ncols=130
#         loss = 10000
#         for (x, y) in process_bar:
#             time.sleep(0.01)
#             process_bar.set_description('Train epoch: {}'.format(epoch + 1))
#             loss -= 1
#             process_bar.set_postfix_str('loss:{:.4f}'.format(loss))


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    loss_epochs = []
    acc_epochs = []
    test_acc_epochs = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        process_bar = tqdm(train_iter)
        for (X, y) in process_bar:
            X = X.to(device)
            y = y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().cpu().item()
            n += y.shape[0]
            batch_count += 1
        test_acc = evaluate_accuracy(test_iter, net)
        loss_epochs.append(train_l_sum / batch_count)
        acc_epochs.append(train_acc_sum / n)
        test_acc_epochs.append(test_acc)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, time %.1f sec'
              % (epoch + 1, train_l_sum / batch_count, train_acc_sum / n, test_acc, time.time() - start))
    save_img(loss_epochs, acc_epochs, test_acc_epochs)


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(net, torch.nn.Module):
                net.eval() # 评估模式, 这会关闭dropout
                acc_sum += (net(X.to(device)).argmax(dim=1) == y.to(device)).float().sum().cpu().item()
                net.train() # 改回训练模式
            else: # 自定义的模型, 3.13节之后不会用到, 不考虑GPU
                if('is_training' in net.__code__.co_varnames): # 如果有is_training这个参数
                    # 将is_training设置成False
                    acc_sum += (net(X, is_training=False).argmax(dim=1) == y).float().sum().item()
                else:
                    acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
            n += y.shape[0]
    return acc_sum / n


def compute_loss(data_iter, net, loss, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for X, y in data_iter:
            X = X.to(device)
            y = y.to(device)
            net.eval() # 评估模式, 这会关闭dropout
            y_hat = net(X)
            l = loss(y_hat, y)
            loss_sum += l
            net.train() # 改回训练模式
            n += y.shape[0]
    return loss_sum / n


def main():
    train_sents, train_labels = load_data(train_path)
    test_sents, test_labels = load_data(test_path)

    # from data_process import analysis_len
    # analysis_len([sent.split() for sent in train_sents])

    x = pd.concat([train_sents, test_sents])
    y = torch.cat((train_labels, test_labels), -1)
    train_sents, test_sents, train_labels, test_labels = train_test_split(x, y, test_size=0.2)

    vocab = get_vocab(x)
    vocab_dic = vocab.get_stoi()

    print('# words in vocab:', len(vocab))
    train_input_file = "/train_input3.npy"
    test_input_file = "/test_input3.npy"

    train_input = preprocess_data(train_sents, vocab_dic, train_input_file)
    test_input = preprocess_data(test_sents, vocab_dic, test_input_file)

    # train_input = get_sents_ids(train_input_file)
    train_set = TensorDataset(train_input, train_labels)
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    # test_input = get_sents_ids(test_input_file)
    test_set = TensorDataset(test_input, test_labels)
    test_loader = DataLoader(test_set, batch_size)

    # 创建网络
    model = RNNClassifier(vocab, embed_size, num_hiddens, num_layers, 5)
    # 加载Glove词向量
    model.embedding.weight.data.copy_(
        load_pretrained_embedding(vocab.get_itos(), glove_vocab))
    model.embedding.weight.requires_grad = False  # 直接加载预训练好的, 所以不需要更新它

    # 要过滤掉不计算梯度的embedding参数
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()), lr=lr)
    loss = nn.CrossEntropyLoss()

    train(train_loader, test_loader, model, loss, optimizer, device, num_epochs)


if __name__ == "__main__":
    main()
