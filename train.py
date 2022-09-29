import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable

from data import *
from net import OctNet
from utils import AverageMeter

file_path = "data_path.csv"
input_height = 543
input_width = 543
output_height = 885
output_width = 512
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])  # editable

model = OctNet()  # editable
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))  # editable
criterion = nn.SmoothL1Loss()  # editable

train_loader = get_data_loader(file_path=file_path,
                               input_height=input_height,
                               input_width=input_width,
                               output_height=output_height,
                               output_width=output_width,
                               transform=transform,
                               batch_size=10)

epochs = 50  # editable
print_freq = 5  # editable

for epoch in range(epochs):
    train_loss = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var, target_val = Variable(input), Variable(target)

        output = model(input_var)

        loss = criterion(output, target_val)
        train_loss.update(loss.data.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % print_freq == 0:
            print('Epoch: [{0}] \t'
                  'Iter: [{1}/{2}]\t'
                  'Train Loss: {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                epoch, i, len(train_loader),
                loss=train_loss))

    print('Finish Epoch: [{0}]\t'
          'Average Train Loss: {loss.avg:.5f}\t'.format(
        epoch, loss=train_loss))
