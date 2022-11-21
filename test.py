import os.path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from skimage.metrics import normalized_root_mse as compare_nrmse
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from net import *

in_channel = 1
out_channel = 46

input_size = 128
output_size = 128

exp_path = "exp"
if not os.path.exists(exp_path):
    os.mkdir(exp_path)

serial_num = input("The Directory Name: ")
data_path = "for_chuangxinshijian/" + serial_num

pred = None
gt = None


class AverageMeter(object):
    def __init__(self):
        self.count = None
        self.sum = None
        self.avg = None
        self.val = None
        self.reset()

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


class OctDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

    def __len__(self):
        return 1

    def __getitem__(self, item):
        files = os.listdir(data_path)
        img = None
        volume_path = None
        for file in files:
            if file.endswith(".tif"):
                img_path = os.path.join(data_path, file)
                img = Image.open(img_path).resize((input_size, input_size))
                img = np.array(img, dtype=np.float32)[:, :, 0]
            elif file.endswith(".fda_OCT_pngs"):
                volume_path = os.path.join(data_path, file)
        images = np.zeros((input_size, input_size, 1), dtype=np.uint8)
        images[:, :, 0] = img
        if self.transform:
            images = self.transform(images)
        volume = np.zeros((out_channel, output_size, output_size))
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
            oct_path = os.path.join(volume_path, oct_path)
            oct = Image.open(oct_path)
            oct = oct.crop((0, 30, 512, 542))
            oct = oct.resize((output_size, output_size))
            oct = np.array(oct, dtype=np.float32)
            volume[i, :, :] = oct
        global gt
        gt = volume
        return images, volume


def get_error_metrics(im_pred, im_gt):
    im_pred = np.array(im_pred).astype(np.float64)
    im_gt = np.array(im_gt).astype(np.float64)
    assert (im_pred.flatten().shape == im_gt.flatten().shape)
    rmse_pred = compare_nrmse(image_true=im_gt, image_test=im_pred)
    ssim_pred = compare_ssim(im_gt, im_pred)
    mse_pred = mean_squared_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    mae_pred = mean_absolute_error(y_true=im_gt.flatten(), y_pred=im_pred.flatten())
    print('mae: {mae_pred:.4f} | mse: {mse_pred:.4f} | rmse: {rmse_pred:.4f} | ssim: {ssim_pred:.4f}'
          .format(mae_pred=mae_pred,
                  mse_pred=mse_pred,
                  rmse_pred=rmse_pred,
                  ssim_pred=ssim_pred))


def images_save(im_pred, im_gt):
    seq = range(pred.shape[0])
    for idx in seq:
        pd = im_pred[idx, :, :]
        gt = im_gt[idx, :, :]
        f = plt.figure()
        f.add_subplot(1, 4, 1)
        plt.imshow(pd, interpolation='none', cmap='gray')
        plt.title("Output")
        plt.axis("off")
        f.add_subplot(1, 4, 2)
        plt.imshow(gt, interpolation='none', cmap='gray')
        plt.title("Target")
        plt.axis("off")
        f.add_subplot(1, 4, 3)
        plt.imshow(gt - pd, interpolation='none', cmap='gray')
        plt.title("Target - Output")
        plt.axis("off")
        f.add_subplot(1, 4, 4)
        plt.imshow(pd - gt, interpolation='none', cmap='gray')
        plt.title("Output - Target")
        plt.axis("off")
        file_path = os.path.join(save_path, 'ImageSlice_{}.png'.format(idx + 1))
        f.savefig(file_path)
        plt.close()


model = Net()
criterion = nn.MSELoss(reduction='mean')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
dataset = OctDataset(transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

model_name = 'model.pth.tar'
ckpt_file = os.path.join(exp_path, 'model/' + model_name)
if os.path.isfile(ckpt_file):
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], False)
    print("=> loaded checkpoint '{}' ".format(ckpt_file))
else:
    print("=> no checkpoint found at '{}'".format(ckpt_file))
    exit(0)

model.eval()
losses = AverageMeter()
pred = np.zeros((1, out_channel, output_size, output_size), dtype=np.float32)
for i, (input, target) in enumerate(loader):
    input_var, target_var = Variable(input), Variable(target)
    output = model(input_var)
    loss = criterion(output, target_var)
    losses.update(loss.data.item(), input.size(0))
    pred[i, :, :, :] = output.data.float()
    print('{0}: [{1}/{2}]\t'
          'Val Loss {loss.val:.5f} ({loss.avg:.5f})\t'
          .format('test', i, len(loader), loss=losses))
print('Average {} Loss: {y:.5f}\t'.format('test', y=losses.avg))

save_path = os.path.join(exp_path, 'result')
if not os.path.exists(save_path):
    os.mkdir(save_path)
file_name = os.path.join(save_path, 'test_prediction.npz')
print("=> saving test prediction results: '{}'".format(file_name))
np.savez(file_name, pred=pred)

loss = losses.avg
pred = pred[0, ...]
get_error_metrics(im_pred=pred, im_gt=gt)
images_save(pred, gt)
