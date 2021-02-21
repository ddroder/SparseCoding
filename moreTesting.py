import h5py
import torchvision.transforms as transforms
import numpy as np 
from fastmril import fastmri 
from fastmril.fastmri.data.subsample import RandomMaskFunc
import matplotlib.pyplot as plt
from fastmril.fastmri.data import transforms as T
import glob
import torch
from skimage.transform import rescale,resize
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
from testLoadData import loadData
all_files=glob.glob("/nvme_ssd/mriData/singlecoil_train/*.h5")
hf=h5py.File("/nvme_ssd/mriData/singlecoil_train/file1002569.h5","r")
print('Keys:', list(hf.keys()))
print('Attrs:', dict(hf.attrs))
volume_kspace = hf['kspace'][()]


def show_slices(data, cmap=None):
    fig = plt.figure()
    plt.imshow(data, cmap=cmap)


def get_tensor_complex_image(filepath):
    hf=h5py.File(filepath)
    volume_kspace=hf['kspace'][()]
    slice_kspace = volume_kspace[20] # Choosing the 20-th slice of this volume
    slice_kspace2 = T.to_tensor(slice_kspace)      # Convert from numpy array to pytorch tensor
    slice_image = T.ifft2(slice_kspace2)           # Apply Inverse Fourier Transform to get the complex image
    slice_image_abs = T.complex_abs(slice_image)   # Compute absolute value to get a real image
    resize1=resize(slice_image_abs,(500,500),anti_aliasing=True)
    # slice_image_abs=slice_image_abs.numpy()
    return resize1

tensorfied_images=[]
for image in all_files:
    c=get_tensor_complex_image(image)
    tensorfied_images.append(c)
thing=loadData(tensorfied_images,500,10)
train_data=[]
for batch_idx, (true_sigs) in enumerate(thing):
    train_data.append(true_sigs)
im_test=train_data[9][0]
print(im_test)
print(type(im_test))
print(im_test.shape)
im_test=im_test.numpy()
im_test2=im_test.reshape(500,500)
plt.imshow(im_test2,cmap='grey')
plt.show()
# print(thing)
# print(len(tensorfied_images))
# print(tensorfied_images[0])

