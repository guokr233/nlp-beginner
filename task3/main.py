from dataset import *
from torch.utils.data import DataLoader
from model import ESIM
import time
import torch.nn as nn

import torchtext.vocab as Vocab
import os
from utils import load_pretrained_embedding

train_path = "./data/snli_1.0_train.jsonl"
test_path = "./data/snli_1.0_dev.jsonl"
max_len = 25
batch_size = 64
vocab = load_vocab(vocab_path="./data/vocab")  # 加载字典
vocab_size = len(vocab)
embed_dim = 300
hidden_size = 300
lr = 0.005
num_epochs = 10
DATA_ROOT = "./data"


""" 设置随机种子 """
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    model = ESIM(vocab_size, embed_dim, hidden_size)
    glove_vocab = Vocab.GloVe(name='6B', dim=embed_dim, cache=os.path.join(DATA_ROOT, "glove"))
    # 加载Glove词向量
    model.embedding.weight.data.copy_(load_pretrained_embedding(vocab.get_itos(), glove_vocab))

    # w2id_dic = vocab.get_stoi()
    train_set = SentPairDataset(train_path, max_len, data_mode="train")
    train_loader = DataLoader(train_set, batch_size, shuffle=True)
    test_set = SentPairDataset(test_path, max_len, data_mode="test")
    test_loader = DataLoader(test_set, batch_size, shuffle=True)


    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=lr, betas=(0.9, 0.999))
    train(test_loader, test_loader, model, loss, optimizer, device, num_epochs, 10)


def train(train_iter, test_iter, net, loss, optimizer, device, num_epochs, max_gradient_norm):
    net = net.to(device)
    print("training on ", device)
    batch_count = 0
    loss_epochs = []
    acc_epochs = []
    test_acc_epochs = []
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        process_bar = tqdm(train_iter)
        for (s1, s1_len, s2, s2_len, labels) in process_bar:
            s1 = s1.to(device)
            s1_len = s1_len.to(device)
            s2 = s2.to(device)
            s2_len = s2_len.to(device)
            s2_len = s2_len.to(device)
            labels = labels.to(device)
            y_hat = net(s1, s2, s1_len, s2_len, device)
            l = loss(y_hat, labels)
            optimizer.zero_grad()
            l.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_gradient_norm)
            optimizer.step()
            train_l_sum += l.cpu().item()
            train_acc_sum += (y_hat.argmax(dim=1) == labels).sum().cpu().item()
            n += labels.shape[0]
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
        # process_bar = tqdm(data_iter)
        # process_bar.set_description("test")
        for _, (s1, s1_len, s2, s2_len, labels) in enumerate(data_iter):
            s1 = s1.to(device)
            s1_len = s1_len.to(device)
            s2 = s2.to(device)
            s2_len = s2_len.to(device)
            labels = labels.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(s1, s2, s1_len, s2_len).argmax(dim=1) == labels.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += labels.shape[0]
    return acc_sum / n


def save_img(loss, acc, test_acc):
    num_epochs = len(loss)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, test_acc, 'r', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.show()
    plt.savefig("acc1.png")

    # plt.plot(epochs, loss, 'r', label='Training loss')
    # # plt.plot(epochs, val_loss, 'b', label='validation loss')
    # plt.title('Training and validation loss')
    # plt.legend()
    # plt.savefig("loss1.png")


if __name__ == "__main__":
    main()