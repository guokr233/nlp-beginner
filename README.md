# NLP-Beginner：自然语言处理入门练习 Report


### 任务一：基于机器学习的文本分类
实现基于logistic/softmax regression的文本分类

#### 1.1 程序流程

1. 建立词表

   读取所有的文本，进行分词，并建立一个词表收录所有的词汇，也可以采用n-gram，即将n个连续的词视作一个单位

2. 提取特征

   对一个句子进行分词，并统计词频，得到一个长度为V的向量，V为词表大小，向量第i个值表示第i个词的词频，该向量即为该句子的特征

3. 模型计算

   输入向量经过一个线性层，再经过一个softmax层，则得到一个在类别上的概率分布，损失函数为 -log P(i)，其中i为真实类别，P(i)为模型对类别i的预测概率

#### 1.2 结果

* 训练集准确度：0.52 
* 测试集准确度：0.49




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
   
4. 后面发现去除停用词会导致性能略有下降，可能是因为在情感分析任务中not之类的词比较重要，如果不去除停用词能再有一到两个点的性能提升

##### 2.1.3 遇到的困难

1. 问题：准确度低于别人的实现结果

   解决：run别人的代码，控制变量比较（模型参数、数据的表示方式、数据的组织），最后发现是因为别人只使用了一小部分数据，且打乱了数据

2. 学习nn.LSTM的使用
    outputs, (h_n, c_n) = nn.LSTM(inputs)
   or
   outputs, (h_n, c_n) = nn.LSTM(inputs, hidden)
   
   * inputs：输入张量，**(seq_len, batch, embed_size)**
   * hidden：自己指定的隐藏变量（可选），LSTM的hidden为包含两个tensor的tuple，分别为h和c，维度为**(1, batch, hidden_size)**
   * outputs： 输出张量，**(seq_len, batch, num_directions \* hidden_size)**，其中双向网络的num_directions为2，单向网络为1
   * h_n：中间向量，**(num_layers \* num_directions, batch, hidden_size)**
   * c_n：记忆单元存储的向量，**(num_layers \* num_directions, batch, hidden_size)**



#### 2.2 CNN提取特征(TextCNN)

##### 2.2.1 模型处理数据的流程

* 嵌入层输入张量维度为(批量大小，词数)
* 嵌入层输出张量维度为(批量大小, 词数, 词向量维度)，词向量维度对应输出通道数，词数对应输出通道宽
* 经过一个卷积层，如卷积核宽为n，长为m，输出张量维度为(批量大小，词数-n+1，词向量维度-m+1)
* 经过一个最大池化层，对每个通道取最大值作为代表，输出张量维度为(批量大小, 词向量维度-m+1)
* 最后经过一个全连接层进行分类

##### 2.2.2 训练结果

测试集准确度0.65左右，与RNN相近![CNN-2glove](https://s2.loli.net/2022/03/05/kG3MyXZJ7wDubxl.png)





### 任务三：基于注意力机制的文本匹配

输入两个句子判断，判断它们之间的关系。参考[ESIM]( https://arxiv.org/pdf/1609.06038v3.pdf)（可以只用LSTM，忽略Tree-LSTM），用双向的注意力机制实现

#### 3.1 论文笔记

##### 3.1.1、模型结构

1. 输入编码（链式LSTM）

   对两个句子进行编码，得到句子的编码

2. 局部推理建模

   * 使用**软性注意力 & 点积模型** 计算两个句子施加注意力后的编码

   * 注意力计算及使用方式

     其中eij表示a第i个词与b第j个词的注意力分数，所有注意力分数经过softmax之后作为权重计算b各个词向量的加权和得到新ai，bi同理

     <img src="https://s2.loli.net/2022/03/07/8rZ4YnhXcRfBDsq.png" alt="image-20220307090354628" style="zoom:67%;" />

     <img src="https://s2.loli.net/2022/03/07/3bjCJkIBg26YnOP.png" alt="image-20220307090332976" style="zoom:67%;" />

   * 将旧ai、新ai、它们的差、哈达玛积（对应元素相乘）拼接

     <img src="https://s2.loli.net/2022/03/07/vJ8HiUnIA9OYxXT.png" alt="image-20220307091027124" style="zoom:67%;" />

3. 推理组合
   1. 使用ma、mb再次经过一个一个全连接层，一个双向LSTM层，均使用ReLU作为激活函数
   1. 分别经过最大池化和平均池化（在seq_len那一维），将结果拼接成一个向量
   1. 最后将结果送入一个多层感知机：一个隐藏层和一个softmax输出层，隐藏层以tanh作为激活函数


##### 3.1.2 训练方式

* 数据：SNLI语料库

  句子对的三种关系：蕴含、无关、矛盾（去除其他类别）

  测试集：development set

* 模型

  * LSTM和字嵌入的隐藏状态维度：300
  * Dropout：应用于所有全连接层
  * 词嵌入向量：300维的Glove 840B向量（训练中会更新）
  * OOV词汇：通过高斯样本初始化

* 训练

  * Adam优化器（第一个动量为0.9，第二个动量为0.999）
  * 初始学习率：0.0004
  * 批量大小为32
  * Tesla K40训练时间：6h

* 评估

  * 准确度
  * ESIM：88%
  * 使用了TreeLSTM的HIM：88.6%

##### 3.1.3 消融实验

重要组件

* 池化层（而不是求和）
* 向量差和哈达玛积
* 使用双向LSTM进行句子编码（而不是全连接层）
* 两个句子的注意力（第二个句子的注意力更重要）

#### 3.2 训练结果

1. 参数设置

   完全使用论文中的模型以及训练参数

2. 训练结果

   * 准确度

     在第11个epoch后测试集准确度最高，为86.2%

     <img src="https://s2.loli.net/2022/03/09/2VEbrXdQkINKGuc.png" alt="myplot" style="zoom: 50%;" />

   * loss

     <img src="https://s2.loli.net/2022/03/09/quN6K78D1t4vePG.png" alt="loss1" style="zoom: 50%;" />

#### 3.3 遇到的困难

问题：模型结构比较复杂

跟踪一个batch在模型中的数据流动，弄清楚每个模块在做什么






### 任务四：基于LSTM+CRF的序列标注

用LSTM+CRF来训练序列标注模型：以Named Entity Recognition为例。

#### 4.1 LSTM-CRF

1. 使用LSTM自动提取特征，得到句子的特征表示，size: (bacth, max_len, num_labels)
2. 使用条件随机场，通过标签的转移概率矩阵P，约束生成的标签符合规范

#### 4.2 训练结果

* 准确率：92 %

* 加权平均

  * 精确率 92 %
  * 召回率 92 %
  * f1 score 91 %

* 直接平均

  * 精确率 69 %
  * 召回率 66 %
  * f1 score 67 % 

* <img src="https://s2.loli.net/2022/03/11/dEzYnUojKRgbH59.png" alt="image-20220311102855157" style="zoom:80%;" />

* 分析

  直接平均的数值低，数量加权平均的数值高

  可能是因为各个标签的分布非常不均衡，作为非实体的“O”标签数量非常多，而"I-MISC"、“B-LOC”只有其百分之一不到，部分该标签被分为“O”标签，导致召回率低。又因数量少，部分其他标签被分为这些标签，就很容易拉低精确率。

  可能的解决方法：从数据入手，增加这些少量标签的数量。



### 任务五：基于神经网络的语言模型

用LSTM、GRU来训练字符级的语言模型，计算困惑度

#### 5.1 训练

RNN语言模型

1. 输入：（batch, max_len），不足最大长度的用0填充
2. 经过嵌入层：（batch, max_len, embed_size）
3. 经过LSTM进行编码，再经过一个全连接层进行分类，分别得到以部分前缀预测下一个字的概率：P(w2 | w1)、P(w3 | w1 w2)、P(w4 | w1w2w3)……P(wn | w1w2w3…wn-1)，长度为max_len-1，原句中后max_len-1个字一一对应为实际标签，两者可计算交叉熵，并完成训练

#### 5.2 结果

* 困惑度

  <img src="https://s2.loli.net/2022/03/14/P1ONdZv7hBsa9by.png" alt="preplexity" style="zoom:72%;" />

* 以日月光华 旦复旦兮为句首生成的诗

  <img src="https://s2.loli.net/2022/03/14/OI2zNfSHZ6JRLYn.png" alt="image-20220313113032285" style="zoom:67%;" />
