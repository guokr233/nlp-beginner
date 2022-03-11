import torch.nn as nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from torchcrf import CRF


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_tags, embed_dim, hidden_dim, max_len, dropout=None):
        super(BiLSTM_CRF, self).__init__()
        self.max_len = max_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, bidirectional=True)
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.dropout_pec = dropout
        if dropout:
            self.dropout = nn.Dropout(dropout)
        self.crf = CRF(num_tags)

    # 由长度列表计算mask矩阵
    def get_mask(self, length_list):
        mask = []
        for length in length_list:
            mask.append([1 for i in range(length)] + [0 for j in range(self.max_len - length)])
        return torch.tensor(mask, dtype=torch.uint8)

    def encode(self, sentences, length_list):
        embeds = self.embedding(sentences)  # (batch, max_len, embed_size)
        if self.dropout_pec:
            embeds = self.dropout(embeds)
        packed_sentences = pack_padded_sequence(embeds, lengths=length_list, batch_first=True, enforce_sorted=False)
        # try:
        #     lstm_out, _ = self.lstm(packed_sentences)
        # except:
        #     print(1)
        lstm_out, _ = self.lstm(packed_sentences)
        result, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=self.max_len)
        if self.dropout_pec:
            result = self.dropout(result)
        feature = self.linear(result)
        return feature

    def loss(self, sentences, length_list, targets):
        # 使用Bi-LSTM对输入进行编码
        encoded_inputs = self.encode(sentences, length_list)
        return -1 * self.crf(torch.transpose(encoded_inputs, 0, 1), torch.transpose(targets, 0, 1),
                        torch.transpose(self.get_mask(length_list), 0, 1))

    def forward(self, sentences, length_list):
        out = self.encode(sentences, length_list)
        return self.crf.decode(torch.transpose(out, 0, 1), torch.transpose(self.get_mask(length_list), 0, 1))

