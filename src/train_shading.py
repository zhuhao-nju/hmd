from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import PIL.Image
import datetime
import pickle

from data_loader import dataloader_shading
from network import shading_Unet
from utility import take_notes, photometricLossgray

# parsing argmument
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=8,
                    help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=30,
                    help='input batch size')
parser.add_argument('--imageSize', type=int, default=[448, 448],
                    help='the height/width of the input image to network')
parser.add_argument('--use_color', type=bool, default=False,
                    help='use photometric color loss or not')
parser.add_argument('--nepoch', type=int, default=20,
                    help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type=bool, default=True,
                    help='enables cuda')
parser.add_argument('--gpu_ids', type=int, default=3,
                    help='which GPU to use')
parser.add_argument('--finetune', default='',
                    help="path to net (to continue training)")
parser.add_argument('--outf', default='./model/snapshots/',
                    help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, default=1234,
                    help='manual seed')
parser.add_argument('--testInterval', type=int, default=5000,
                    help='test interval')
parser.add_argument('--prvInterval', type=int, default=1,
                    help='preview interval')
parser.add_argument('--shlInterval', type=int, default=5,
                    help='show loss interval')
parser.add_argument('--saveModelEpcoh', type=int, default=10,
                    help='save model interval')
opt = parser.parse_args()
print(opt)


if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)
    print("New directory for output is built: " + opt.outf)

# generate random seed
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
print("Random Seed: ", opt.manualSeed)

# get current time
c_time = datetime.datetime.now()
time_string = "%s-%02d:%02d:%02d" % (c_time.date(), c_time.hour, c_time.minute, c_time.second)

# write log file
f_log = open(opt.outf + "log.txt", 'w')
f_log.write("time: %s\r\n" % time_string)
for arg in vars(opt):
    f_log.write("%s: %s\r\n" % (arg, getattr(opt, arg)))

# remind to use cuda
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# prepare for cuda
device = torch.device("cuda:" + str(opt.gpu_ids) if opt.cuda else "cpu")

# get dataset class
dataset = dataloader_shading(manual_seed=opt.manualSeed,
                             transform=transforms.Compose([transforms.ToTensor()]),
                             use_color=opt.use_color)
assert dataset
dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=opt.batchSize,
                                         shuffle=True,
                                         num_workers=int(opt.workers))

# transfer model to device (GPU or CPU)
net_shading = shading_Unet(in_channels=dataset[0][0].shape[0],  init_weights=True).train().to(device)

if opt.finetune != '':
    net_shading.load_state_dict(torch.load(opt.finetune))

criterion = nn.L1Loss()

# define the optimizer
optimizer = optim.Adam(net_shading.parameters(), lr=opt.lr, betas=(opt.beta1, 0.99), weight_decay=0.0005)

# prepare for loss saving
batch_num = dataloader.dataset.num / dataloader.batch_size
take_notes("===Loss===|==Epoch==|==Iter==", opt.outf + "loss.txt", create_file=True)

# start training
for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):

        # load data
        input_src = data[0].to(device).float()
        input_mask = data[1].to(device).float()
        depth_gt = data[2].to(device).float()
        depth_sm = data[3].to(device).float()
        depth_diff = data[4].to(device).float()

        if opt.use_color:
            input_src_gray = data[5].to(device).float()
            input_alebedo = data[6].to(device).float()
            input_alebedo_gray = data[7].to(device).float()
            light_est = data[8].to(device).float()

        # forward and backward propagate
        optimizer.zero_grad()

        pred_diff = net_shading(input_src, input_mask)
        depth_loss = F.mse_loss(pred_diff, depth_diff)

        if opt.use_color:
            color_loss, _ = photometricLossgray(input_src_gray, depth_sm + pred_diff*0.1, input_alebedo_gray, input_mask,
                                                light_est, device, K=[400.0, 400.0, 224.0, 224.0], thres=30)
        else:
            color_loss = 0.0

        loss = depth_loss + 100.0 * color_loss

        loss.backward()
        optimizer.step()

        # show loss, save loss
        if i % opt.shlInterval == 0:
            print("step: %d/%d, loss: %f, depth loss: %f, color loss: %f "\
                  % (i + epoch * batch_num, opt.nepoch * batch_num, loss, depth_loss, color_loss))
            take_notes("\r\n%10f %7d %8d" \
                       % (loss, epoch, i), opt.outf + "loss.txt")

    if epoch == (opt.nepoch - 1):
        torch.save(net_shading.state_dict(), "%spretrained_shading.pth" % opt.outf)
        break

    # save parameters for each epoch
    if epoch % opt.saveModelEpcoh == 0:
        torch.save(net_shading.state_dict(),
                   "%sshading_epoch_%d.pth" % (opt.outf, epoch))

print("Done, final model saved to %spretrained_shading.pth" % opt.outf)

