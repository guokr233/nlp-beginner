import torch
from utils import *
from dataset import NERDataset, get_loader
from tqdm import tqdm
from model import BiLSTM_CRF
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from torch.optim import Adam
import numpy as np

pad_item = "<PAD>"
pad_id = 0
nof_item = "<UNK>"    # OOV词
nof_id = 1
n_classes = 5
embed_size = 100
hidden_size = 100
dropout = 0.5
train_path = "./data/train.bmes"
test_path = "./data/test.bmes"
batch_size = 64
lr = 0.001
num_epochs = 10
max_gradient_norm=10
""" 设置随机种子 """
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

x_train, y_train = load_data(train_path)
x_test, y_test = load_data(test_path)

word2id, tag2id = get_dict(x_train+x_test, y_train+y_test, pad_id, nof_id, pad_item, nof_item)
vocab_size = len(word2id)
num_tags = len(tag2id)
print("vocab_size: ", vocab_size)
print(tag2id)
print("训练集样本数：", len(x_train))
print("测试集样本数：", len(x_test))

length_list = [len(sent.split()) for sent in x_test+x_train]
# max_len = max(length_list)
# analysis_len(x_train+x_test)
max_len = 45    # 超过99%的句子长度


model = BiLSTM_CRF(vocab_size, num_tags, embed_size, hidden_size, max_len)
model.to(device)
train_loader = get_loader(x_train, y_train, word2id, tag2id, max_len, nof_id, pad_id, batch_size)
test_loader = get_loader(x_test, y_test, word2id, tag2id, max_len, nof_id, pad_id, batch_size)
optimizer = Adam(model.parameters(), lr)


def train(train_iter, test_iter, model, optimizer, device, num_epochs, max_gradient_norm=10):
    model = model.to(device)
    print("training on ", device)
    model.train()
    for epoch in range(num_epochs):
        process_bar = tqdm(train_iter)
        for x, y, lens in process_bar:
            x = x.to(device)
            y = y.to(device)
            lens = lens.to(device)
            optimizer.zero_grad()
            loss = model.loss(x, lens, y)
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
            loss.backward()
            optimizer.step()
        test(test_iter, model)
        torch.save(model.state_dict(), 'params.pkl')


def test(test_iter, model):
    aver_loss = 0
    preds, labels = [], []
    process_bar = tqdm(test_iter)
    for x_test, y_test, lens_test in process_bar:
        model.eval()
        predict = model(x_test, lens_test)
        # CRF
        loss = model.loss(x_test, lens_test, y_test)
        aver_loss += loss.item()
        # 统计非0的，也就是真实标签的长度
        leng = []
        for i in y_test.cpu():
            tmp = []
            for j in i:
                if j.item() > 0:
                    tmp.append(j.item())
            leng.append(tmp)
        for index, i in enumerate(predict):
            preds += i[:len(leng[index])]
        for index, i in enumerate(y_test.tolist()):
            labels += i[:len(leng[index])]
    aver_loss /= (len(test_iter) * 64)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    report = classification_report(labels, preds)
    print("loss: ", aver_loss)
    print(report)


# train(test_loader, test_loader, model, optimizer, device, num_epochs, max_gradient_norm)
train(train_loader, test_loader, model, optimizer, device, num_epochs, max_gradient_norm)



