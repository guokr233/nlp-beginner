from torch.utils.data import Dataset
import pandas as pd


class PhraseDataset(Dataset):
    def __init__(self, data_path):
        """
        :param csv_file: csv文件的路径
        :param root_dir: 图像的文件夹路径
        :param transform: 可选的transform
        """
        print("Start loading data...")
        data = pd.read_csv(data_path, sep="\t")
        print("Load data over...")
        print("data.shape: ", data.shape)  # (156060, 4)
        print("data.keys: ", data.keys())  # ['PhraseId', 'SentenceId', 'Phrase', 'Sentiment']
        # 提取句子与标签的列
        x = data["Phrase"]
        y = data["Sentiment"]
        print("x.shape：", x.shape)
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        sample = (self.x[idx], self.y[idx])
        return sample
