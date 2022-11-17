#!/bin/bash
#训练命令 初始epoch为200 可以parse指定 其他parse可以通过options文件夹下看
# python train.py --dataroot ./datasets/try/ --name try_cyclegan --model cycle_gan --pool_size 50 --no_dropout --gpu_ids 0
#test命令
#python test.py --dataroot ./datasets/try/ --name test_cyclegan --model cycle_gan --phase test --no_dropout

#简单运行该指令进入环境
conda activate pytorch-CycleGAN-and-pix2pix