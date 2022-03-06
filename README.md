# NLP-Beginner：自然语言处理入门练习 Report


### 任务一：基于机器学习的文本分类
实现基于logistic/softmax regression的文本分类




### 任务二：基于深度学习的文本分类
熟悉Pytorch，用Pytorch重写《任务一》，实现CNN、RNN的文本分类；
#### 2.1 RNN提取特征
##### 2.1.1 程序流程

1. 对所有文本通过nltk进行清洗、分词，统计得到词典，**词典大小为14533**
2. 利用词典将句子转成词序号的列表，长的截断，短的补0
3. 进入embbeding层转成(seq_len, batch_size, embed_size)的词嵌入张量
4. 通过双向LSTM层得到提取的文本特征（两个方向上output张量的拼接）
5. 经过全连接层进行分类

##### 2.1.2 训练结果

1. 不使用glove词向量：测试集准确度0.65左右

   ![glove-update-0.2-2](https://s2.loli.net/2022/03/05/ig3TpqX2hdLH7a1.png)

3. 使用glove词向量但不更新参数：测试集准确度0.65左右，与1相近

   ![glove-update-0.2-2](https://s2.loli.net/2022/03/05/JbwcdghQzTqUsp5.png)

4. 使用词向量并更新参数：测试集准确度0.66左右，略高于1、2

   ![glove-update-0.2](https://s2.loli.net/2022/03/05/WC5RQEaynv24dKz.png)

##### 2.1.3 遇到的困难

1. 问题：准确度低于别人的实现结果

   解决：run别人的代码，控制变量比较（模型参数、数据的表示方式、数据的组织），最后发现是因为别人只使用了一小部分数据，且打乱了数据

2. 问题：nn.LSTM的使用

   解决：输出的三个张量的含义与维度

   * output： 输出张量，**(seq_len, batch, num_directions \* hidden_size)**，其中双向网络的num_directions为2，单向网络为1
   * h_n：中间向量，**(num_layers \* num_directions, batch, hidden_size)**
   * c_n：记忆单元存储的向量，**(num_layers \* num_directions, batch, hidden_size)**



#### 2.2 CNN提取特征(TextCNN)

##### 2.2.1 模型处理数据的流程

* 嵌入层输入张量维度为(批量大小，词数)
* 嵌入层输出张量维度为(批量大小, 词数, 词向量维度)，词向量维度对应输出通道数，词数对应输出通道宽
* 经过一个卷积层，如卷积核宽为n，长为m，输出张量维度为(批量大小，词数-n-1，词向量维度-m+1)
* 经过一个最大池化层，对每个通道取最大值作为代表，输出张量维度为(批量大小, 词向量维度-m+1)
* 最后经过一个全连接层进行分类

##### 2.2.2 训练结果

测试集准确度0.65左右，与RNN相近![CNN-2glove](https://s2.loli.net/2022/03/05/kG3MyXZJ7wDubxl.png)





### 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现。


### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。


### 任务五：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

