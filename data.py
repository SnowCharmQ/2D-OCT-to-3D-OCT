import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class Oct3dDataset(Dataset):
    def __init__(self, file_path, num_views, input_height, input_width,
                 output_height, output_width, transform=None):
        self.df = pd.read_csv(file_path)
        self.num_views = num_views
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        projs = np.zeros((self.input_width, self.input_height, self.num_views), dtype=np.int8)

        proj_path = self.df.iloc[idx]['2D_data_path']
        proj = Image.open(proj_path)
        projs[:, :, 0] = np.array(proj)

        if self.transform:
            projs = self.transform(projs)

        vol_path = self.df.iloc[idx]['3D_data_path']
        volume = np.load(vol_path)

        volume = volume - np.min(volume)
        volume = volume / np.max(volume)
        assert ((np.max(volume) - 1.0 < 1e-3) and (np.min(volume) < 1e-3))

        volume = torch.from_numpy(volume)

        return projs, volume
