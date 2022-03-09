import torch.nn as nn
import numpy as np
import torch
from utils import sort_by_seq_lens, masked_softmax, weighted_sum, get_mask, replace_masked


class ESIM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, embeddings=None,
                 padding_idx=0, dropout=0.5, num_classes=3):
        """
        Args:
            vocab_size: 词表大小
            embed_dim: 词嵌入维度
            hidden_size: 隐藏层维度
            embeddings: 预训练的词向量，若为None则随机初始化
            padding_idx:  padding token的序号，默认为0
            dropout: 全连接层的dropout比例，默认为0.5
            num_classes: 输出层的类别，默认为3
            device: 默认为cpu
        """
        super(ESIM, self).__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        # 嵌入层
        self.word_embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=padding_idx)
        if embeddings:
            # 加载预训练的词嵌入
            embvecs, embwords = embeddings
            self.word_embedding.weight.data.copy_(torch.from_numpy(np.asarray(embvecs)))
        # self.word_embedding.weight.requires_grad = False

        if self.dropout:
            self.rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)

        # Encoder，由双向LSTM实现
        self.encoding = Seq2SeqEncoder(nn.LSTM, self.embed_dim,
                                       self.hidden_size, bidirectional=True)

        # 软性注意力，使用点积模型
        self.attention = SoftmaxAttention()
        # 使用ReLU作为激活函数的全连接层
        self.projection = nn.Sequential(nn.Linear(4 * 2 * self.hidden_size,
                                                  self.hidden_size), nn.ReLU())
        # 双向LSTM
        self.composition = Seq2SeqEncoder(nn.LSTM, self.hidden_size,
                                          self.hidden_size, bidirectional=True)
        # 分类器
        # 一个隐藏层，以tanh为激活函数，进行dropout
        self.classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                            nn.Linear(2 * 4 * self.hidden_size, self.hidden_size),
                                            nn.Tanh(),
                                            nn.Dropout(p=self.dropout),
                                            nn.Linear(self.hidden_size, self.num_classes))
        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                sent1_ids,
                sent2_ids,
                sent1_lens=None,
                sent2_lens=None,
                device='cpu'
                ):
        """
        :param sent1_ids:   句子1，维度 (batch_size, max_len).
        :param sent2_ids:   句子2，维度 (batch_size, max_len).
        :param sent1_lens:  句子1batch中每个句子的长度，一维张量
        :param sent2_lens:  句子2batch中每个句子的长度，一维张量
        :param device:      运行设备
        :return:
            logits: A tensor of size (batch, num_classes) containing the logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing the probabilities of each output class in the model.

        """


        # premises = text_a_ids
        # premises_lengths = q1_lens
        # hypotheses = text_b_ids
        # hypotheses_lengths = q2_lens

        # 计算两个句子的mask
        # mask's size：(batch, max_seq_len)
        # 其中max_seq_len是批量中最长句子的长度
        sent1_mask = get_mask(sent1_ids, sent1_lens).to(device)
        sent2_mask = get_mask(sent2_ids, sent2_lens).to(device)

        # 得到词嵌入矩阵
        # embedded_sent's size：(batch_size, max_len, embed_size)
        embedded_sent1 = self.word_embedding(sent1_ids)
        embedded_sent2 = self.word_embedding(sent2_ids)

        # 对RNN的输入进行dropout，丢掉一部分输入（置为0）
        if self.dropout:
            embedded_sent1 = self.rnn_dropout(embedded_sent1)
            embedded_sent2 = self.rnn_dropout(embedded_sent2)
        # 使用双向LSTM对词嵌入进行初步编码
        # encoded_sent's size : (batch_size, max_seq_len, embed_size * 2)
        encoded_sent1 = self.encoding(embedded_sent1, sent1_lens)
        encoded_sent2 = self.encoding(embedded_sent2, sent2_lens)

        # 使用软性注意力计算得到注意力加权后的编码
        # mask1：在相似度矩阵中将pad元素所在的位置全部置为0，再经过softmax，再将pad元素所在的位置全部置为0
        # mask2：在将词嵌入进行加权后，将pad元素所在的位置全部置为0
        # attended_sent's size : (batch_size, max_seq_len, embed_size * 2)
        attended_sent1, attended_sent2 = self.attention(encoded_sent1, sent1_mask,
                                                        encoded_sent2, sent2_mask)
        # 将原编码、注意力编码、它们的差、它们的哈达玛积进行拼接
        # enhanced_sent's size : (batch_size, max_seq_len, embed_size * 8)
        enhanced_sent1 = torch.cat([encoded_sent1,
                                    attended_sent1,
                                    encoded_sent1 - attended_sent1,
                                    encoded_sent1 * attended_sent1],
                                    dim=-1)
        enhanced_sent2 = torch.cat([encoded_sent2,
                                    attended_sent2,
                                    encoded_sent2 - attended_sent2,
                                    encoded_sent2 * attended_sent2],
                                    dim=-1)

        # 全连接映射层，减少参数
        # projected_sent's size : (batch_size, max_seq_len, embed_size)
        projected_sent1 = self.projection(enhanced_sent1)
        projected_sent2 = self.projection(enhanced_sent2)

        # 对RNN的输入进行dropout
        if self.dropout:
            projected_sent1 = self.rnn_dropout(projected_sent1)
            projected_sent2 = self.rnn_dropout(projected_sent2)

        # 经过双向LSTM
        # size : (batch_size, max_seq_len, 2 * embed_size)
        v_ai = self.composition(projected_sent1, sent1_lens)
        v_bj = self.composition(projected_sent2, sent2_lens)

        # 平均池化
        # size: (batch_size, 2 * embed_size)
        v_a_avg = torch.sum(v_ai * sent1_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(sent1_mask, dim=1, keepdim=True)
        v_b_avg = torch.sum(v_bj * sent2_mask.unsqueeze(1)
                            .transpose(2, 1), dim=1) \
                  / torch.sum(sent2_mask, dim=1, keepdim=True)
        # 最大池化
        v_a_max, _ = replace_masked(v_ai, sent1_mask, -1e7).max(dim=1)
        v_b_max, _ = replace_masked(v_bj, sent2_mask, -1e7).max(dim=1)
        # 拼接
        # v's size : (batch_size, 2 * 4 * embed_size)
        v = torch.cat([v_a_avg, v_a_max, v_b_avg, v_b_max], dim=1)

        # 输出
        # size : (batch_size, num_inputs)
        logits = self.classification(v)
        # probabilities = nn.functional.softmax(logits, dim=-1)

        return logits


class RNNDropout(nn.Dropout):
    def forward(self, sequences_batch):
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        return dropout_mask.unsqueeze(1) * sequences_batch


class Seq2SeqEncoder(nn.Module):
    def __init__(self,
                 rnn_type,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 bias=True,
                 dropout=0.0,
                 bidirectional=False):
        assert issubclass(rnn_type, nn.RNNBase), \
            "rnn_type must be a class inheriting from torch.nn.RNNBase"

        super(Seq2SeqEncoder, self).__init__()

        self.rnn_type = rnn_type
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional

        self._encoder = rnn_type(input_size,
                                 hidden_size,
                                 num_layers=num_layers,
                                 bias=bias,
                                 batch_first=True,
                                 dropout=dropout,
                                 bidirectional=bidirectional)

    def forward(self, sequences_batch, sequences_lengths):
        sorted_batch, sorted_lengths, _, restoration_idx = \
            sort_by_seq_lens(sequences_batch, sequences_lengths)

        packed_batch = nn.utils.rnn.pack_padded_sequence(sorted_batch,
                                                         sorted_lengths,
                                                         batch_first=True)

        outputs, _ = self._encoder(packed_batch, None)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,
                                                      batch_first=True)
        reordered_outputs = outputs.index_select(0, restoration_idx)

        return reordered_outputs


class SoftmaxAttention(nn.Module):
    """
    Attention layer
    接受RNN对句子的编码作为输入，输出软性注意力
    首先计算两个句子向量的点积，softmax后的结果作为权重
    句子2向量的加权和将用来表示句子1
    反之，句子1向量的加权和将用来表示句子2
    """
    def forward(self,
                premise_batch,
                premise_mask,
                hypothesis_batch,
                hypothesis_mask):
        """
        Args:
            premise_batch: A batch of sequences of vectors representing the
                premises in some NLI task. The batch is assumed to have the
                size (batch, max_seq_len, vector_dim).
            premise_mask: 用于在计算注意力时，忽略掉padding的0
            hypothesis_batch: A batch of sequences of vectors representing the
                hypotheses in some NLI task. The batch is assumed to have the
                size (batch, max_seq_len, vector_dim).
            hypothesis_mask: 用于在计算注意力时，忽略掉padding的0
        """
        # Dot product between premises and hypotheses in each sequence of
        # the batch.
        # batch (max_seq_len1, vector_dim) 乘 batch (vector_dim, max_seq_len2) = (batch, max_seq_len1, max_seq_len2)
        similarity_matrix = premise_batch.bmm(hypothesis_batch.transpose(2, 1)
                                              .contiguous())

        # Softmax attention 权重
        # mask1：在相似度矩阵中将pad元素所在的位置全部置为0，再经过softmax，再将pad元素所在的位置全部置为0
        prem_hyp_attn = masked_softmax(similarity_matrix, hypothesis_mask)
        hyp_prem_attn = masked_softmax(similarity_matrix.transpose(1, 2)
                                       .contiguous(),
                                       premise_mask)

        # Weighted sums of the hypotheses for the the premises attention,
        # and vice-versa for the attention of the hypotheses.
        # mask2：在将词嵌入进行加权后，将pad元素所在的位置全部置为0

        attended_premises = weighted_sum(hypothesis_batch,
                                         prem_hyp_attn,
                                         premise_mask)
        attended_hypotheses = weighted_sum(premise_batch,
                                           hyp_prem_attn,
                                           hypothesis_mask)

        return attended_premises, attended_hypotheses


def _init_esim_weights(module):
    """
    Initialise the weights of the ESIM model.
    """
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.kaiming_normal_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2 * hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.kaiming_normal_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2 * hidden_size)] = 1.0