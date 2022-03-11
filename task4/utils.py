import matplotlib.pyplot as plt
import torch


def pad(x, max_len, pad_id):
    if len(x) > max_len:
        return x[:max_len]
    else:
        return x + [pad_id] * (max_len - len(x))


def sents2id(vocab_dic, words, max_len, nof_id, pad_id=0):
    vec = []
    for word in words:
        try:
            vec.append(vocab_dic[word])
        except KeyError:
            vec.append(nof_id)
    return pad(vec, max_len, pad_id)


def analysis_len(sents):
    sents_len = [len(sent.split()) for sent in sents]
    sents_len = sorted(sents_len)
    nums = len(sents_len)
    print("70%: ", sents_len[int(0.7 * nums)])
    print("80%: ", sents_len[int(0.8 * nums)])
    print("90%: ", sents_len[int(0.9 * nums)])
    print("95%: ", sents_len[int(0.95 * nums)])
    print("99%: ", sents_len[int(0.99 * nums)])
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.figure(figsize=(30, 12), dpi=100)
    plt.subplot(2, 3, 2)
    plt.title("句子长度分布")
    plt.hist(sents_len, bins=list(range(0, max(sents_len) + 1, 1)))
    plt.xlabel('句子长度')
    plt.ylabel('句子数量')
    """ title 累计分布"""
    plt.subplot(2, 3, 5)
    plt.title('累计分布图')
    plt.hist(sents_len, bins=list(range(0, max(sents_len) + 1, 1)), cumulative=True)
    plt.xlabel('句子长度')
    plt.ylabel('累计比例(%)')

    plt.savefig("sent_len.png")


def load_data(path, length=-1):
    sentences = []  # 每个 str 都是一个 word，List[str] 表示一个句子，List[List[str]] 表示一堆句子
    labels = []

    with open(path, 'r', encoding='UTF-8') as f:
        sent = []
        sent_labels = []
        for line in f:
            line = line.strip()  # 去除换行
            if not line:  # 空白行：两个句子的分隔符
                sentences.append(' '.join(sent))
                labels.append(' '.join(sent_labels))
                sent = []
                sent_labels = []
            else:
                split_result = line.split()
                sent.append(split_result[0])
                sent_labels.append(split_result[1])

    return sentences[:length], labels[:length]


def build_dict(x):
    ret = []
    for i in x:
        ret += [j for j in i.split()]
    return list(set(ret))


def get_dict(x, y, pad_id, nof_id, pad_item, nof_item):
    items_x = build_dict(x)
    items_y = build_dict(y)
    items_x.insert(pad_id, pad_item)
    items_x.insert(nof_id, nof_item)
    items_y.insert(pad_id, pad_item)
    word2idx = {items_x[i]: i for i in range(len(items_x))}
    num_tags = len(items_y)
    tag2id = {items_y[i]: i for i in range(num_tags)}
    tag2id["<START>"] = num_tags
    tag2id["<STOP>"] = num_tags + 1

    return word2idx, tag2id


def save_img(loss, acc, test_acc):
    num_epochs = len(loss)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, acc, 'b', label='Training accuracy')
    plt.plot(epochs, test_acc, 'r', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.show()
    plt.savefig("acc1.png")

    plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.plot(epochs, val_loss, 'b', label='validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("loss1.png")


def evaluate_accuracy(data_iter, net, device=None):
    if device is None and isinstance(net, torch.nn.Module):
        # 如果没指定device就使用net的device
        device = list(net.parameters())[0].device
    acc_sum, n = 0.0, 0
    with torch.no_grad():
        # process_bar = tqdm(data_iter)
        # process_bar.set_description("test")
        for _, (s1, s1_len, s2, s2_len, labels) in enumerate(data_iter):
            s1 = s1.to(device)
            s1_len = s1_len.to(device)
            s2 = s2.to(device)
            s2_len = s2_len.to(device)
            labels = labels.to(device)
            net.eval()  # 评估模式, 这会关闭dropout
            acc_sum += (net(s1, s2, s1_len, s2_len).argmax(dim=1) == labels.to(device)).float().sum().cpu().item()
            net.train()  # 改回训练模式
            n += labels.shape[0]
    return acc_sum / n
