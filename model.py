#this is a testing file to test
#if the encoder works
#it also generates images of the encoding.
import torch
from torchsummary import summary
import os
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch import Tensor
import cv2
import torch.nn.functional as F
from loadImDat import loadData
from UTILS.class_dict import dictionary
import torchvision
from keras.datasets import mnist
from loadImDat import scaleTo01
import matplotlib.pyplot as plt

data_size=32**2
code_size=1*data_size
device="gpu"
model_dir="/nvme_ssd/bensCode/SparseCoding/models/model3.pt"
net=torch.load(model_dir)
net.eval()
print(net)
(train_X, train_y), (test_X, test_y) = mnist.load_data()

trainload,testload=loadData("mnist",32,10)
train_data=[]
for batch_idx, (true_sigs, im_labels) in enumerate(trainload):
    train_data.append(true_sigs)
im_test=train_data[9][0]
im_test2=im_test.reshape(32,32)
og_image=plt.imshow(im_test2)
# plt.show()
im_test=im_test.to('cuda:0')
yhat=net(im_test)
yhat=yhat.cpu().data.numpy()
print(yhat.shape)
yhat=yhat.reshape(32,32)
# plt.show()
fig,axs=plt.subplots(1,2,sharey='row')
axs[0].imshow(im_test2)
axs[0].set_title("Original Image (Image passed to model)")
axs[1].imshow(yhat)
axs[1].set_title("Encoded Image (Model output)")
plt.subplots_adjust(wspace=.8)
plt.show()