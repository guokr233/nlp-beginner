### NLP-Beginner-Task1-基于机器学习的文本分类



#### 一、数据集
1. 数据结构：
   1. PhraseId：短语ID
   2. SentenceId：句子ID，一个Sentence对应多个Phrase，为什么？
   3. Phrase：短语内容
   4. Sentiment：情感标签
      * 0 - negative 
      * 1 - somewhat negative 
      * 2 - neutral 
      * 3 - somewhat positive 
      * 4 - positive
2. pandas读取tsv数据 over
3. 分词、预处理 over
#### 二、文本特征
1. BOW
   1. 创建语料库的词表，实现存储与恢复 over 
   2. 将一个句子表示成BOW向量 over
   3. 存储和读取BOW向量 over
2. N-Gram
   1. 用sklearn获得TF-IDF计算的N-Gram文本特征
#### 三、分类算法
1. 推理过程
2. loss、训练过程
3. 评估过程




训练结果
1. 
   1. 分类器：逻辑回归
   2. loss：对数loss
   3. N-gram：（1，1）
   4. 训练集准确度： 0.5166093410572049 
   5. 测试集准确度：0.49379113646336115
2. 
   1. 是