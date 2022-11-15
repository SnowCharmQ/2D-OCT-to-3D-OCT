import os
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def generate_data_path(path: str):
    tif_path, volume_path = None, None
    for filename in os.listdir(path):
        suffix = os.path.splitext(filename)[-1]
        if suffix == '.tif':
            tif_path = os.path.join(path, filename)
        elif suffix == '.fda_OCT_pngs':
            images_path = os.path.join(path, filename)
            volume = np.zeros((46, 128, 128), dtype=np.int64)
            numbers = []
            for i in range(128):
                if (i + 3) % 3 == 0:
                    numbers.append(i)
            numbers.append(49)
            numbers.append(55)
            numbers.append(59)
            numbers.sort()
            for i in range(len(numbers)):
                oct_path = f"octscan_{numbers[i] + 1}.png"
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


def get_file_path(epoch, i, j, k, info="output", dic_path="img"):
    current_path = os.getcwd()
    if not os.path.exists(dic_path):
        os.mkdir(dic_path)
    img_path = "%s_epoch%s_iter%s_batch%s_no%s.png" % (info, epoch, i, j, k)
    file_path = os.path.join(current_path, dic_path, img_path)
    return file_path


def save_volumetric_images(epoch, i, output, default='output'):
    output = output.detach().numpy()
    for j in range(len(output)):
        for k in range(len(output[j])):
            img = output[j][k]
            Image.fromarray(img).convert("L").save(get_file_path(epoch, i, j, k, default))
    print("Saved %s images in epoch %d iter %d" % (default, epoch, i))


def save_diff_images(epoch, i, output, target, plane=0):
    output = output.detach().numpy()
    target = target.detach().numpy()
    for batch in range(len(output)):
        output_batch = output[batch]
        target_batch = target[batch]
        seq = range(output_batch.shape[plane])
        for idx in seq:
            if plane == 0:
                pd = output_batch[idx, :, :]
                gt = target_batch[idx, :, :]
            elif plane == 1:
                pd = output_batch[:, idx, :]
                gt = target_batch[:, idx, :]
            elif plane == 2:
                pd = output_batch[:, :, idx]
                gt = output_batch[:, :, idx]
            else:
                assert False
            f = plt.figure()
            f.add_subplot(1, 4, 1)
            plt.imshow(pd)
            plt.title("Output")
            plt.axis("off")
            f.add_subplot(1, 4, 2)
            plt.imshow(gt)
            plt.title("Target")
            plt.axis("off")
            f.add_subplot(1, 4, 3)
            plt.imshow(gt - pd)
            plt.title("Target - Output")
            plt.axis("off")
            f.add_subplot(1, 4, 4)
            plt.imshow(pd - gt)
            plt.title("Output - Target")
            plt.axis("off")
            file_path = get_file_path(epoch, i, batch, idx, "diff", "diff")
            f.savefig(file_path)
            plt.close()
    print("Saved difference images in epoch %d iter %d" % (epoch, i))


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
