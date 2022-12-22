from torch.autograd import Variable
from torchvision import transforms

from net import *
from data import *
from utils import *
from data_processor.generator import generate

file_path = "data_path.csv"
if not os.path.exists(file_path):
    generate()

out_channels = 46
input_height = 128
input_width = 128
output_height = 128
output_width = 128

model = Net()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_loader = get_test_loader(file_path=file_path,
                              input_height=input_height,
                              input_width=input_width,
                              output_height=output_height,
                              output_width=output_width,
                              transform=transform,
                              batch_size=1,
                              proportion=0.8)

model_name = 'model_train.pth.tar'
ckpt_file = os.path.join("model", model_name)
if os.path.isfile(ckpt_file):
    checkpoint = torch.load(ckpt_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'], False)
    print("=> loaded checkpoint '{}' ".format(ckpt_file))
else:
    print("=> no checkpoint found at '{}'".format(ckpt_file))
    exit(0)

model.eval()
print("Start testing...")
for i, (input, target) in enumerate(test_loader):
    input_var, target_var = Variable(input), Variable(target)
    output = model(input_var)
    pd = output.data.float()
    gt = target.data.float()
    save_comparison_images(pd, gt, mode="test", iter=i)
    mse_pd, mae_pd, rmse_pd, psnr_pd, ssim_pd = get_error_metrics(pd, gt)
    print('mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
          ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f} in test {iter}'
          .format(mse_pred=mse_pd,
                  mae_pred=mae_pd,
                  rmse_pred=rmse_pd,
                  psnr_pred=psnr_pd,
                  ssim_pred=ssim_pd,
                  iter=i))
