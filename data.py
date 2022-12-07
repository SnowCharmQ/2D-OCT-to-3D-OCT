import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class TrainDataset(Dataset):
    def __init__(self, file_path, input_height, input_width,
                 output_height, output_width, transform=None, proportion=0.8, val_proportion=0.7):
        self.df = pd.read_csv(file_path)
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.transform = transform
        self.proportion = proportion
        self.val_proportion = val_proportion

    def __len__(self):
        return math.floor(len(self.df) * self.proportion * self.val_proportion)

    def __getitem__(self, idx):
        projs = np.zeros((self.input_width, self.input_height, 1), dtype=np.float32)

        proj_path = self.df.iloc[idx]['2D_data_path']
        proj = Image.open(proj_path).resize((self.input_height, self.input_width))
        proj = np.array(proj, dtype=np.float32)
        projs[:, :, 0] = proj[:, :, 0]

        if self.transform:
            projs = self.transform(projs)

        vol_path = self.df.iloc[idx]['3D_data_path']

        volume = np.load(vol_path)
        # volume = volume - np.min(volume)
        # volume = volume / np.max(volume)
        volume = torch.from_numpy(volume).float()

        return projs, volume


def get_train_loader(file_path, input_height, input_width, output_height, output_width,
                     transform, batch_size, proportion=0.8, val_proportion=0.7):
    dataset = TrainDataset(file_path, input_height, input_width, output_height,
                           output_width, transform, proportion, val_proportion)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    return loader


class ValDataset(Dataset):
    def __init__(self, file_path, input_height, input_width,
                 output_height, output_width, transform=None,
                 proportion=0.8, val_proportion=0.7):
        self.df = pd.read_csv(file_path)
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.transform = transform
        self.proportion = proportion
        self.val_proportion = val_proportion

    def __len__(self):
        return math.floor(len(self.df) * self.proportion) - \
               math.floor(len(self.df) * self.proportion * self.val_proportion)

    def __getitem__(self, idx):
        idx = idx + math.floor(len(self.df) * self.proportion * self.val_proportion)
        projs = np.zeros((self.input_width, self.input_height, 1), dtype=np.float32)

        proj_path = self.df.iloc[idx]['2D_data_path']
        proj = Image.open(proj_path).resize((self.input_height, self.input_width))
        projs[:, :, 0] = np.array(proj, dtype=np.float32)[:, :, 0]

        if self.transform:
            projs = self.transform(projs)

        vol_path = self.df.iloc[idx]['3D_data_path']

        volume = np.load(vol_path)
        volume = volume - np.min(volume)
        volume = volume / np.max(volume)
        volume = torch.from_numpy(volume).float()

        return projs, volume


def get_val_loader(file_path, input_height, input_width, output_height, output_width,
                   transform, batch_size, proportion=0.8, val_proportion=0.7):
    dataset = ValDataset(file_path, input_height, input_width, output_height, output_width,
                         transform, proportion, val_proportion)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, pin_memory=False)
    return loader


class TestDataset(Dataset):
    def __init__(self, file_path, input_height, input_width,
                 output_height, output_width, transform=None, proportion=0.8):
        self.df = pd.read_csv(file_path)
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.transform = transform
        self.proportion = proportion

    def __len__(self):
        return len(self.df) - math.floor(len(self.df) * self.proportion)

    def __getitem__(self, idx):
        idx = idx + math.floor(len(self.df) * self.proportion)
        projs = np.zeros((self.input_width, self.input_height, 1), dtype=np.float32)

        proj_path = self.df.iloc[idx]['2D_data_path']
        proj = Image.open(proj_path).resize((self.input_height, self.input_width))
        projs[:, :, 0] = np.array(proj, dtype=np.float32)[:, :, 0]

        if self.transform:
            projs = self.transform(projs)

        vol_path = self.df.iloc[idx]['3D_data_path']

        volume = np.load(vol_path)
        volume = volume - np.min(volume)
        volume = volume / np.max(volume)
        volume = torch.from_numpy(volume).float()

        return projs, volume


def get_test_loader(file_path, input_height, input_width, output_height, output_width,
                    transform, batch_size=1, proportion=0.8):
    dataset = TestDataset(file_path, input_height, input_width, output_height, output_width, transform, proportion)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader
