import gc
import time

from torchvision import transforms
from torch.autograd import Variable
from data_processor.cleaner import clean
from data_processor.generator import generate

from data import *
from net import *
from utils import *

clean()
file_path = "data_path.csv"
if not os.path.exists(file_path):
    generate()

device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

# input_height = 543
# input_width = 543
# output_height = 885
# output_width = 512
input_height = 128
input_width = 128
output_height = 128
output_width = 128
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
model = ReconNet()
# model = nn.DataParallel(model, device_ids=device_ids)
# model = model.cuda(device=device_ids[0])
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, betas=(0.5, 0.999))
criterion = nn.MSELoss(reduction="mean")
criterion = criterion.cuda(device=device_ids[0])

train_loader = get_data_loader(file_path=file_path,
                               input_height=input_height,
                               input_width=input_width,
                               output_height=output_height,
                               output_width=output_width,
                               transform=transform,
                               batch_size=4)

epochs = 50
print_freq = 5
best_loss = 1e5
print("Start training...")
for epoch in range(epochs):
    s = time.time()
    train_loss = AverageMeter()
    model.train()

    for i, (input, target) in enumerate(train_loader):
        input_var, target_val = Variable(input), Variable(target)
        # input_var = input_var.cuda(device=device_ids[0])
        # target_val = target_val.cuda(device=device_ids[0])

        output = model(input_var).float()

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
            save_volumetric_images(epoch, i, target, 'target')
            save_volumetric_images(epoch, i, output)
            save_diff_images(epoch, i, output, target)

    print('Finish Epoch: [{0}]\t'
          'Average Train Loss: {loss.avg:.5f}\t'.format(
        epoch, loss=train_loss))
    e = time.time()
    print("Time used in one epoch: ", (e - s))
    if train_loss.avg < best_loss:
        best_loss = train_loss.avg
        state = {'epoch': epoch + 1,
                 'model': 'ReconNet',
                 'state_dict': model.state_dict(),
                 'loss': best_loss,
                 'optimizer': optimizer.state_dict()
                 }
        filename = os.path.join(os.getcwd(), "model", "model.pth.tar")
        torch.save(state, filename)
        print("! Save the best model in epoch: {}, the current loss: {}".format(epoch, best_loss))
    gc.collect()
    torch.cuda.empty_cache()
