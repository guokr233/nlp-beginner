from torch import nn


# CNN分类器模型
class CNNClassifier(nn.Module):
    def __init__(self, model_path=None):
        # 从下载好的文件夹中加载预训练模型
        # self.embedding = nn.Embedding()
        # self.cnn = n
        self.dense = nn.Linear(768, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs_ids, attention_mask):
        bert_output = self.bert(inputs_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:,0,:]       # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)     # 用于分类的全连接层
        return linear_output

    def predict(self, inputs_ids, attention_mask):
        bert_output = self.bert(inputs_ids, attention_mask=attention_mask)
        bert_cls_hidden_state = bert_output[0][:,0,:]       # 提取[CLS]对应的隐藏状态
        linear_output = self.dense(bert_cls_hidden_state)     # 用于分类的全连接层
        return self.softmax(linear_output)
