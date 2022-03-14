from torch import nn
import torch

batch_size = 32
max_len = 150
vocab_size = 1314
embedding_dim = 111
hidden_dim = 100
num_layers = 1

RNN = nn.RNN(embedding_dim, hidden_dim, num_layers)
LSTM = nn.LSTM(embedding_dim, hidden_dim, num_layers)

inputs = torch.randn((batch_size, max_len, embedding_dim)).transpose(0, 1)
RNN_hidden = torch.randn((1, batch_size, hidden_dim))
LSTM_hidden = (torch.randn((1, batch_size, hidden_dim)), torch.randn((1, batch_size, hidden_dim)))

print("over")
