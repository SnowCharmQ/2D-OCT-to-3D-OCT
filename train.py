import gc
import time
# from torchstat import stat
from torchvision import transforms
from torch.autograd import Variable
from data_processor import *
from data import *
from net import *
from utils import *

file_path = "data_path.csv"
clean()
if not os.path.exists(file_path):
    generate()

cuda_id = 0
device_name = "cuda:{}".format(cuda_id)
device = torch.device(device_name)
device_ids = [cuda_id]

out_channels = 128
input_height = 128
input_width = 128
output_height = 128
output_width = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
model = Net()
# stat(model, (1, 128, 128))
model = nn.DataParallel(model, device_ids=device_ids)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 80, gamma=0.1, last_epoch=-1)
criterion = nn.MSELoss(reduction='mean')
criterion = criterion.to(device)

patch = (1, input_height // 2 ** 4, input_width // 2 ** 4)
discriminator = Discriminator()
discriminator = discriminator.to(device)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.001, betas=(0.5, 0.999))
criterion_GAN = torch.nn.MSELoss()
criterion_GAN = criterion_GAN.to(device)

train_loader = get_train_loader(file_path=file_path,
                                input_height=input_height,
                                input_width=input_width,
                                output_height=output_height,
                                output_width=output_width,
                                transform=transform,
                                batch_size=4,
                                proportion=0.85,
                                val_proportion=0.7)
val_loader = get_val_loader(file_path=file_path,
                            input_height=input_height,
                            input_width=input_width,
                            output_height=output_height,
                            output_width=output_width,
                            transform=transform,
                            batch_size=1,
                            proportion=0.85,
                            val_proportion=0.7)
test_loader = get_test_loader(file_path=file_path,
                              input_height=input_height,
                              input_width=input_width,
                              output_height=output_height,
                              output_width=output_width,
                              transform=transform,
                              batch_size=1,
                              proportion=0.85)

epochs = 500
print_freq = 4
best_train_loss = 1e7
best_val_loss = 1e7
delta = 100

print("Start training...")
for epoch in range(epochs):
    print("Start training in epoch {}".format(epoch))
    s = time.time()
    train_loss = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var, target_var = Variable(input), Variable(target)
        input_var = input_var.to(device)
        target_var = target_var.to(device)

        # Adversarial ground truths
        valid = np.ones((input_var.size(0), *patch))
        valid = torch.from_numpy(valid)
        valid = Variable(valid, requires_grad=False)
        valid = valid.view(-1, 1, 1, 8, 8)
        valid = valid.to(device)
        valid = valid.to(torch.float32)
        fake = np.zeros((input_var.size(0), *patch))
        fake = torch.from_numpy(fake)
        fake = Variable(fake, requires_grad=False)
        fake = fake.view(-1, 1, 1, 8, 8)
        fake = fake.to(device)
        fake = fake.to(torch.float32)

        output = model(input_var).float()
        output_gan = output.view(-1, 1, out_channels, input_height, input_width)
        target_var = target_var.view(-1, 1, out_channels, input_height, input_width)

        # ---------------------
        #  Train Generator
        # ---------------------

        optimizer.zero_grad()

        # GAN loss
        pred_fake = discriminator(output_gan)
        pred_fake = pred_fake.to(torch.float32)
        loss_GAN = criterion_GAN(pred_fake, valid)  # MSELoss

        loss = criterion(output_gan, target_var) * delta + loss_GAN
        loss = loss.to(torch.float32)

        train_loss.update(loss.data.item(), input.size(0))

        loss.backward()
        optimizer.step()
        # scheduler.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # # Real loss
        pred_real = discriminator(target_var)
        loss_real = criterion_GAN(pred_real, valid)

        # # Fake loss
        pred_fake = discriminator(output_gan.detach())
        loss_fake = criterion_GAN(pred_fake, fake)

        # # Total loss
        loss_D = 0.5 * (loss_real + loss_fake)
        loss_D.backward()
        optimizer_D.step()

        if i % print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}] \t'
                  'Iter: [{1}/{2}]\t'
                  'Train Loss: {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader) - 1,
                loss=train_loss))
            pd = output.data.float().cpu()
            gt = target.data.float().cpu()
            save_comparison_images(pd, gt, mode="train", epoch=epoch, iter=i)

    print('Finish Epoch: [{0}]\t'
          'Average Train Loss: {loss.avg:.5f}\t'.format(
        epoch, loss=train_loss))
    e = time.time()
    print("Time used in training one epoch: ", (e - s))

    if train_loss.avg < best_train_loss:
        best_train_loss = train_loss.avg
        state = {'epoch': epoch + 1,
                 'model': 'ReconNet',
                 'state_dict': model.state_dict(),
                 'loss': best_train_loss,
                 'optimizer': optimizer.state_dict()}
        if not os.path.exists("model"):
            os.mkdir("model")
        filename = os.path.join(os.getcwd(), "model", "model_train.pth.tar")
        torch.save(state, filename)
        torch.save(model.state_dict(), "model/model_train.pkl")
        print("! Save the best model in epoch: {}, the current train loss: {}".format(epoch, best_train_loss))

    model.eval()
    print("Start validating in epoch {}".format(epoch))
    s = time.time()
    mse, mae, rmse, psnr, ssim = 0, 0, 0, 0, 0
    cnt = 0
    for i, (input, target) in enumerate(val_loader):
        input_var, target_var = Variable(input), Variable(target)
        # input_var = input_var.to(device)
        output = model(input_var)
        pd = output.data.float().cpu()
        gt = target.data.float().cpu()
        save_comparison_images(pd, gt, mode="validation", epoch=epoch, iter=i)
        mse_pd, mae_pd, rmse_pd, psnr_pd, ssim_pd = get_error_metrics(pd, gt)
        mse += mse_pd
        mae += mae_pd
        rmse += rmse_pd
        psnr += psnr_pd
        ssim += ssim_pd
        cnt += 1
    mse /= cnt
    mse /= out_channels
    mae /= cnt
    mae /= out_channels
    rmse /= cnt
    rmse /= out_channels
    psnr /= cnt
    psnr /= out_channels
    ssim /= cnt
    ssim /= out_channels
    print('Average mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
          ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f}'
          .format(mse_pred=mse,
                  mae_pred=mae,
                  rmse_pred=rmse,
                  psnr_pred=psnr,
                  ssim_pred=ssim))
    if mse < best_val_loss:
        best_val_loss = mse
        state = {'epoch': epoch + 1,
                 'model': 'ReconNet',
                 'state_dict': model.state_dict(),
                 'loss': best_val_loss,
                 'optimizer': optimizer.state_dict()}
        if not os.path.exists("model"):
            os.mkdir("model")
        filename = os.path.join(os.getcwd(), "model", "model_val.pth.tar")
        torch.save(state, filename)
        torch.save(model.state_dict(), "model/model_val.pkl")
    e = time.time()
    print("Time used in validating one epoch: ", (e - s))
    gc.collect()

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
    mse_pd /= out_channels
    mae_pd /= out_channels
    rmse_pd /= out_channels
    psnr_pd /= out_channels
    ssim_pd /= out_channels
    print('Average mse: {mse_pred:.4f} | mae: {mae_pred:.4f} | rmse: {rmse_pred:.4f} |'
          ' psnr: {psnr_pred:.4f} | ssim: {ssim_pred:.4f} in test {iter}'
          .format(mse_pred=mse_pd,
                  mae_pred=mae_pd,
                  rmse_pred=rmse_pd,
                  psnr_pred=psnr_pd,
                  ssim_pred=ssim_pd,
                  iter=i))
