import torch
from utils import read_data, sent2id, pad, poetry2id, init_hidden, generate
from torch.utils.data import DataLoader
from torch import optim, nn
import numpy as np
from tqdm import tqdm
from model import PoetModel


data_path = "./data/poetryFromTang.txt"
# data_path = "./data/poetryFromTang2.txt"
max_len = 160
pad_id = 0
pad_item = "<PAD>"
drop_out = 0.2
embed_dim = 100
hidden_dim = 100
num_layers = 1

batch_size = 32
lr = 0.0005
num_epochs = 20
max_gradient_norm = 10
""" 设置随机种子 """
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


poetry_list = read_data(data_path, nums=2000)
chars = [c for poetry in poetry_list for c in poetry]
chars.insert(0, "<PAD>")
chars = set(chars)
char2id = {char: id for id, char in enumerate(chars)}
# char2id['<START>'] = len(char2id)
# char2id['<EOS>'] = len(char2id)
# char2id['</s>'] = len(char2id)
id2char = {id: char for char, id in list(char2id.items())}
vocab_size = len(char2id)
print("vocab_size: ", vocab_size)

data = poetry2id(poetry_list, char2id, max_len, pad_id)
data = torch.tensor(data)
data_loader = DataLoader(data, shuffle=True, batch_size=batch_size)


def train(data_loader, model, num_epochs, optimizer, criterion, char2id, id2char, device):
    # test(data_loader, model, criterion, device)
    loss_list, preplexity_list = [], []
    loss, preplexity = test(data_loader, model, criterion, device)
    loss_list.append(loss)
    preplexity_list.append(preplexity)
    for epoch in range(num_epochs):
        # head = '我爱琴宝'
        # model.eval()
        # model.generate(head, max_len=14)
        # print("over.")
        model.train()
        num_batch = 0
        process_bar = tqdm(data_loader)
        for x in process_bar:
            # 1.处理数据
            # x: (batch_size,max_len) ==> (max_len, batch_size)
            x = x.long().transpose(1, 0).contiguous()
            x = x.to(device)
            optimizer.zero_grad()
            # input, target: (max_len-1, batch_size)
            input_, target = x[:-1, :], x[1:, :]
            target = target.view(-1)
            # 初始化hidden为(c0, h0): ((layer_num， batch_size, hidden_dim)，(layer_num， batch_size, hidden_dim)）
            hidden = init_hidden(num_layers, x.size()[1], hidden_dim)

            # 2.前向计算
            # print(input.size(), hidden[0].size(), target.size())
            output, _ = model(input_, hidden)
            # output, _ = model(input_)
            loss = criterion(output, target)  # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)
            # 反向计算梯度
            loss.backward()
            # 权重更新
            optimizer.step()
            num_batch += 1
        # print('epoch: %d' % (epoch))
        loss, preplexity = test(data_loader, model, criterion, device)
        loss_list.append(loss)
        preplexity_list.append(preplexity)
        head = '日月光华旦复旦兮'
        model.eval()
        model.generate(head, max_len=7)
        # generate(model, head, char2id, id2char, 14)
    print(loss_list, preplexity_list)


def test(data_loader, model, criterion, device):
    model.eval()
    avg_loss = 0
    process_bar = tqdm(data_loader)
    bacth_count = 0
    with torch.autograd.no_grad():
        for x in process_bar:
            # 1.处理数据
            x = x.long().transpose(1, 0).contiguous()
            x = x.to(device)
            input_, target = x[:-1, :], x[1:, :]
            target = target.view(-1)
            hidden = init_hidden(num_layers, x.size()[1], hidden_dim)
            output, _ = model(input_, hidden)
            # output, _ = model(input_)
            loss = criterion(output, target)  # output:(max_len*batch_size,vocab_size), target:(max_len*batch_size)
            avg_loss += loss.cpu().item()
            bacth_count += 1
        avg_loss = avg_loss / bacth_count
        perplexity = pow(2, avg_loss)
    print("test loss: ", avg_loss)
    print("perplexity: ", perplexity)
    return avg_loss, perplexity


model = PoetModel(vocab_size, embed_dim,hidden_dim, num_layers, drop_out, char2id, id2char)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()
train(data_loader, model, num_epochs, optimizer, criterion, char2id, id2char, device)

