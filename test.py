from torch.autograd import Variable
from torchvision import transforms

from net import *
from data import *
from utils import *
from data_processor import *

file_path = "data_path.csv"
if not os.path.exists(file_path):
    generate()

device = torch.device("cuda:0")
device_ids = [0]

out_channels = 46
input_height = 128
input_width = 128
output_height = 128
output_width = 128

model = Net()
model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)

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

model_name = 'model.pkl'
model_file = os.path.join("model", model_name)
if os.path.isfile(model_file):
    model.load_state_dict(torch.load(model_file))
else:
    print("=> no checkpoint found at '{}'".format(model_file))
    exit(0)

if not os.path.exists('test'):
    os.mkdir('test')
log_name = model_name.split('.')[0] + '_test.txt'
log_name = os.path.join('test', log_name)
f = open(log_name, 'w')

model.eval()
print("Start testing...")
for i, (input, target) in enumerate(test_loader):
    input_var, target_var = Variable(input), Variable(target)
    input_var = input_var.to(device)
    output = model(input_var)
    pd = output.data.float().cpu()
    gt = target.data.float().cpu()
    save_comparison_images(pd, gt, mode="test", iter=i)
    mse_pd, mae_pd, rmse_pd, psnr_pd, ssim_pd = get_error_metrics(pd, gt)
    f.write('mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
            ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f} in test {iter}\n'
            .format(mse_pred=mse_pd,
                    mae_pred=mae_pd,
                    rmse_pred=rmse_pd,
                    psnr_pred=psnr_pd,
                    ssim_pred=ssim_pd,
                    iter=i))
    mse_pd /= out_channels
    mae_pd /= out_channels
    rmse_pd /= out_channels
    psnr_pd /= out_channels
    ssim_pd /= out_channels
    f.write('Average mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
            ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f} in test {iter}\n'
            .format(mse_pred=mse_pd,
                    mae_pred=mae_pd,
                    rmse_pred=rmse_pd,
                    psnr_pred=psnr_pd,
                    ssim_pred=ssim_pd,
                    iter=i))
    print("Finish iter {}".format(i))
print("Finish testing")
f.close()
