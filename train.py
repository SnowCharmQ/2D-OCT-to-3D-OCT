import gc
import os
import random
import time

import numpy as np
# from torchstat import stat
from torchvision import transforms
from torch.autograd import Variable
from data_processor.cleaner import clean
from data_processor.generator import generate

from data import *
from net import *
from utils import *

log_name = str(time.time()) + ".log"
file = open(log_name, 'w')
file_path = "data_path.csv"
exp_path = 'exp'
if not os.path.exists(exp_path):
    os.mkdir(exp_path)
clean()
if not os.path.exists(file_path):
    generate()

device = torch.device("cuda")
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

out_channels = 46
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999), weight_decay=0.01)
criterion = nn.MSELoss(reduction='mean')
criterion = criterion.to(device)

train_loader = get_train_loader(file_path=file_path,
                                input_height=input_height,
                                input_width=input_width,
                                output_height=output_height,
                                output_width=output_width,
                                transform=transform,
                                batch_size=4,
                                proportion=0.8,
                                val_proportion=0.7)
val_loader = get_val_loader(file_path=file_path,
                            input_height=input_height,
                            input_width=input_width,
                            output_height=output_height,
                            output_width=output_width,
                            transform=transform,
                            batch_size=1,
                            proportion=0.8,
                            val_proportion=0.7)
test_loader = get_test_loader(file_path=file_path,
                              input_height=input_height,
                              input_width=input_width,
                              output_height=output_height,
                              output_width=output_width,
                              transform=transform,
                              batch_size=1,
                              proportion=0.8)

epochs = 300
print_freq = 10
best_loss = 1e5
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

        output = model(input_var).float()

        loss = criterion(output, target_var)
        train_loss.update(loss.data.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0 or i == len(train_loader) - 1:
            print('Epoch: [{0}] \t'
                  'Iter: [{1}/{2}]\t'
                  'Train Loss: {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader) - 1,
                loss=train_loss))
            # save_volumetric_images(epoch, i, target, 'target')
            # save_volumetric_images(epoch, i, output)
            # save_diff_images(epoch, i, output, target)

    print('Finish Epoch: [{0}]\t'
          'Average Train Loss: {loss.avg:.5f}\t'.format(
        epoch, loss=train_loss))
    e = time.time()
    print("Time used in training one epoch: ", (e - s))
    if train_loss.avg < best_loss:
        best_loss = train_loss.avg
        state = {'epoch': epoch + 1,
                 'model': 'ReconNet',
                 'state_dict': model.state_dict(),
                 'loss': best_loss,
                 'optimizer': optimizer.state_dict()
                 }
        if not os.path.exists("model"):
            os.mkdir("model")
        filename = os.path.join(os.getcwd(), "model", "model.pth.tar")
        torch.save(state, filename)
        print("! Save the best model in epoch: {}, the current loss: {}".format(epoch, best_loss))

    model.eval()
    print("Start validating in epoch {}".format(epoch))
    s = time.time()
    test_size = len(val_loader)
    for i, (input, target) in enumerate(val_loader):
        input_var, target_var = Variable(input), Variable(target)
        # target_var = target_var.to(device)
        output = model(input_var)
        pd = output.data.float()
        gt = target.data.float()
        save_path = os.path.join(exp_path, 'result' + str(i))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
    e = time.time()
    print("Time used in validating one epoch: ", (e - s))

    gc.collect()

model.eval()
print("Start testing...")
for i, (input, target) in enumerate(test_loader):
    input_var, target_var = Variable(input), Variable(target)
    output = model(input_var)
    pd = output.data.float()
    gt = target.data.float()
file.close()
