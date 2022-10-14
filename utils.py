import os
import numpy as np
from PIL import Image


def generate_data_path(path: str):
    tif_path, volume_path = None, None
    for filename in os.listdir(path):
        suffix = os.path.splitext(filename)[-1]
        if suffix == '.tif':
            tif_path = os.path.join(path, filename)
        elif suffix == '.fda_OCT_pngs':
            images_path = os.path.join(path, filename)
            volume = np.zeros((46, 128, 128), dtype=np.int64)
            for i in range(46):
                oct_path = f"octscan_{i + 1}.png"
                oct_path = os.path.join(images_path, oct_path)
                img = Image.open(oct_path)
                img = img.crop((0, 30, 512, 542))
                img = img.resize((128, 128))
                img = np.array(img, dtype=np.float32)
                volume[i, :, :] = img
            volume_path = os.path.join(path, "volume.npy")
            np.save(volume_path, volume)
    assert tif_path is not None
    assert volume_path is not None
    return tif_path, volume_path


def clean_data_npy(path: str):
    for filename in os.listdir(path):
        if filename == 'volume.npy':
            filename = os.path.join(path, filename)
            os.remove(filename)


def save_volumetric_images(epoch, i, output):
    def get_file_path(j, k):
        current_path = os.getcwd()
        dic_path = "img"
        if not os.path.exists(dic_path):
            os.mkdir(dic_path)
        img_path = "epoch%s_iter%s_batch%s_no%s.png" % (epoch, i, j, k)
        file_path = os.path.join(current_path, dic_path, img_path)
        return file_path

    output = output.detach().numpy()
    for j in range(len(output)):
        for k in range(len(output[j])):
            img = output[j][k]
            Image.fromarray(img).convert("L").save(get_file_path(j, k))
    print("Saved images")


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
