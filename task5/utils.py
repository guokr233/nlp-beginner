import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt


def init_hidden(num_layers, batch_size, hidden_dim):
    return (Variable(torch.zeros(num_layers, batch_size, hidden_dim)),
            Variable(torch.zeros(num_layers, batch_size, hidden_dim)))


def read_data(data_path, nums=-1):
    with open(data_path, "r") as f:
        data = f.readlines()
        aPoem = ""
        poem_list = []
        for line in data:
            if line != "\n":
                aPoem += line.strip()
            else:
                poem_list.append(aPoem)
                aPoem = ""
    return poem_list[:nums]


def sent2id(sent, char2id):
    # 按字分隔，并加上开始和结束符号
    # sent = ['<START>'] + list(sent) + ['<EOS>']
    sent = list(sent)
    return [char2id[char] for char in sent]


def pad(x, max_len, pad_id):
    if len(x) > max_len:
        return x[:max_len]
    else:
        return x + [pad_id] * (max_len - len(x))


def poetry2id(poem_list, char2id, max_len, pad_id):
    return [pad(sent2id(poem, char2id), max_len, pad_id) for poem in poem_list]


def generate(model, heads, char2id, id2char, max_length):
    for start in heads:
        x = torch.tensor(char2id[start])
        predicts = model.predict(x, max_length)
        sentence = [start] + [id2char[i] for i in predicts]
        sentence.insert(len(sentence)//2, ",")
        sentence.append(".")
        print(''.join(sentence))



from zhconv import convert
import json
def read_json(data_path):
    list = []
    with open(data_path, 'r') as load_f:
        load_dict = json.load(load_f)
        for poetry in load_dict:
            list.append(convert("".join(poetry["paragraphs"]), "zh-cn"))
    return list


def write_txt(path_list):
    poetry_list = []
    for path in path_list:
        poetry_list += read_json(path)
    with open("data/poetryFromTang2.txt", "w") as f:
        for poetry in poetry_list:
            f.writelines(poetry)
            f.writelines("\n\n")


def save_img(loss, preplexity):
    num_epochs = len(loss)
    epochs = range(1, num_epochs + 1)
    plt.plot(epochs, preplexity, 'r', label='preplexity')
    plt.title('preplexity')
    plt.legend()
    plt.savefig("preplexity.png")
    plt.show()

    # plt.plot(epochs, loss, 'r', label='Training loss')
    # plt.title('Training loss')
    # # plt.figure()
    # plt.legend()
    # plt.savefig("loss.png")
    # plt.show()


# path_list = ["./data/poet.tang." + str(1000 * i) + ".json" for i in range(10)]
# write_txt(path_list=path_list)

loss = [7.770876884460449, 7.220289866129558, 6.682760556538899, 6.2008302211761475, 5.628471453984578, 5.217902104059855, 4.84281325340271, 4.4436759154001875, 4.298888285954793, 4.20650068918864, 3.938958764076233, 3.7528829177220664, 3.8232789834340415, 3.6556848287582397, 3.3797752062479653, 3.3755651712417603, 3.452743411064148, 3.3326046069463096, 3.3803631067276, 3.2448936303456626, 2.993857979774475]
preplexity = [218.40724206127672, 149.11585639988078, 102.73333373442928, 73.55901321415355, 49.46963857112224, 37.217315814688874, 28.696706284527757, 21.761044724724606, 19.68313730752981, 18.462175891387872, 15.337152634471314, 13.48125515301065, 14.155384020415712, 12.602908643883493, 10.409112819638914, 10.378781495905615, 10.949122987356116, 10.07427844822551, 10.413355413723327, 9.480043179830721, 7.9660138044376945]
save_img(loss, preplexity)