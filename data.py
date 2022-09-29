import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class Oct3dDataset(Dataset):
    def __init__(self, file_path, input_height, input_width,
                 output_height, output_width, transform=None):
        self.df = pd.read_csv(file_path)
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        projs = np.zeros((self.input_width, self.input_height, 1), dtype=np.int8)

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


def get_data_loader(file_path, input_height, input_width, output_height, output_width,
                    transform, batch_size, num_workers=None):
    dataset = Oct3dDataset(file_path, input_height, input_width, output_height, output_width, transform)
    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader
