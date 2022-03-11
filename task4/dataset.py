import torch
from torch.utils.data import Dataset, DataLoader
from utils import sents2id


class NERDataset(Dataset):
    def __init__(self, x, y, length_list):
        self.x = x
        self.y = y
        self.length_list = length_list

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.length_list[index]

    def __len__(self):
        return len(self.x)


def get_loader(sents, labels, word2idx, tag2id, max_len, nof_id, pad_id, batch_size):
    inputs = [sents2id(word2idx, sent.split(), max_len, nof_id, pad_id)
              for sent in sents]
    targets = [sents2id(tag2id, v.split(), max_len, nof_id, pad_id)
               for v in labels]
    lens = torch.tensor([min(len(sent.split()), max_len) for sent in sents])
    inputs = torch.tensor(inputs)
    targets = torch.tensor(targets)
    dataset = NERDataset(inputs, targets, lens)
    # dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    return dataloader

