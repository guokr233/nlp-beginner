import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from copy import copy


class PoetModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, char2id, id2char):
        super(PoetModel, self).__init__()
        self.char2id = char2id
        self.id2char = id2char
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=num_layers)
        self.ff = nn.Linear(self.hidden_dim, vocab_size)
        self.dropout_perc = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, hidden):
        seq_len, batch_size = inputs.size()
        # 将one-hot形式的input在嵌入矩阵中转换成嵌入向量，torch.Size([length, batch_size, embedding_size])
        embeds = self.embeddings(inputs)
        if self.dropout_perc:
            embeds = self.dropout(embeds)

        # output (max_len, batch_size, hidden_dim]), 每一个step的输出
        output, hidden = self.lstm(embeds, hidden)

        # 经过线性层，relu激活层 先转换成（max_len*batch_size, 256)维度，再过线性层（length, vocab_size)
        output = F.relu(self.ff(output.view(seq_len*batch_size, -1)))

        # 输出最终结果，与hidden结果
        return output, hidden

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)),
                Variable(torch.zeros(self.num_layers, batch_size, self.hidden_dim)))

    def predict_step(self, start, hidden=None):
        x = self.embeddings(start).unsqueeze(0).view(1, 1, -1)
        if hidden is None:
            hidden = self.init_hidden(1)
        x, hidden = self.lstm(x, hidden)
        out = F.log_softmax(self.ff(x.view(1, -1)), dim=1)
        out = F.softmax(out)
        return out.tolist()[:10], hidden

    def predict(self, start, max_length):
        x = self.embeddings(start).unsqueeze(0).view(1, 1, -1)
        hidden = self.init_hidden(1)
        lstm_out = copy(x)

        for i in range(max_length-1):
            x, hidden = self.lstm(x, hidden)
            lstm_out = torch.cat([lstm_out, x])
        out = F.log_softmax(self.ff(lstm_out.view(max_length, -1)), dim=1)
        out = torch.argmax(out, dim=1)
        return out.tolist()

    def generate(self, heads, max_len, n_gram=7):
        n_gram = max_len * 2
        black_list = [self.char2id['，'], self.char2id['。'], self.char2id["；"]]
        ngram_list = []
        idx = 0
        for start in heads:
            hidden = None
            sentence = start
            ngram_list.append(self.char2id[start])
            for i in range(max_len-1):
                x = torch.tensor([self.char2id[start]])
                prediction, hidden = self.predict_step(x, hidden)
                prediction = prediction[0]
                zip_p = [[i, prediction[i]] for i in range(len(prediction))]
                sorted_p = sorted(zip_p, key=lambda nums: nums[1], reverse=True)[:100]
                filtered_p = [p for p in sorted_p if p[0] not in black_list]
                filtered_p = [p for p in filtered_p if p[0] not in ngram_list]
                x = filtered_p[0][0]
                # if start == "我" and i == 3:
                #     x = self.char2id["雨"]
                sentence += self.id2char[x]
                ngram_list.append(x)
                if len(ngram_list) > n_gram:
                    ngram_list.pop(0)

            if idx % 2 == 0:
                print(sentence + "，", end="")
            else:
                print(sentence + "。")
            idx += 1
