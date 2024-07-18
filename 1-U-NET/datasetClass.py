import torch
import os
from torchvision import transforms
from PIL import Image


class DatasetClassEgomotion(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.source = dataset['source']
        self.target = dataset['target']

        self.source = self.source.unsqueeze(1)
        self.target = self.target.unsqueeze(1)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, item):

        return self.source[item], self.target[item]
