#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Load a dataset, optionally breaking each image into patches.
Note: this is a slow way to process patches; instead use
  data creator to save a patchitized dataset.
        
@author: Benjamin Cowen, 7 Feb 2019
@contact: ben.cowen@nyu.edu, bencowen.com
"""

# TORCH
import os
import torch
import torchvision
import torchvision.transforms as transforms
import math as math

class scaleTo01(object):
    """
    scales one image to [0,1]
    """
    def __init__(self):
        pass
    def __call__(self, img):
        # set to [0, img.max()]
        new_im = img - img.min()
        maxVal = new_im.abs().max()
        if maxVal != 0:
            new_im = new_im/maxVal
        return new_im


# PATCHITIZE
class get_patches(object):
    """
    Extracts **NON-OVERLAPPING** patches from each image.
    """
    #TODO: it's wayyyy faster if we don't have to do this *EVERY TIME* an image
    #      is called. Should just do this once-and-for-all then put it in
    #      a new "patchitized" data loader....?

    def __init__(self,patch_size):
        """
        patch_size: number of pixels in one side of square patch
                     (square patches only)
        """
        #TODO add argument hv_cutoff: discard patches with pixel variance below this threshold
        #            (not implemented yet 3/15/2018)....

        self.patch_size = patch_size
        
    def __call__(self, img): #, variance_cutoff=float('inf')
        """
        img: image tensor of size (D,m,n)
        """
        pSz = self.patch_size
        D  = img.size(0)          # channels
        m  = img.size(1)          # rows
        n  = img.size(2)          # columns

        # If imsize = patchSize: DONE
        if m==pSz and m==n:
            return img

        # Otherwise, setup output tensor:
        Np  =  math.floor(m/pSz) # number of patches per dimension
        ppI = Np**2                     # patches per image

        if D==0:
            patches = torch.Tensor(ppI, pSz, pSz)
        else:
            patches = torch.Tensor(ppI, D, pSz, pSz)
            
        # Now, extract all the (non-overlapping!!!!!!!!!) patches
        next_patch = -1
        for i in range(Np):
            for j in range(Np):
                next_patch += 1
                cSel = i*pSz  # column select
                rSel = j*pSz  # row select
                patches[next_patch]= (img.narrow(1,rSel,pSz).narrow(2,cSel,pSz)).clone()
        return patches
    

# VECTORIZE
class vectorizeIm(object):
    """
    Vectorizes all patches in one image.
    """
    def __init__(self):
        pass
    def __call__(self, img):
        return img
        if img.dim()==3:  # regular (works WITHOUT using "get_patches")
            m = img.size(1)
            n = img.size(2)
            return img.resize_(m*n)
        elif img.dim()==4: # patchitized images
            Np = img.size(0)
            m = img.size(2)
            n = img.size(3)
            return img.resize_(Np,m*n)
            
def fixBsz(bsz,nppi):
    """
    Fix batchsize w.r.t. the number of PATCHES per image
    """
    if bsz<1:
        bsz = 1
    return int(bsz)

##################################################################
## LOAD DATA (uses above classes)
##################################################################
def loadData(arrayData, patchSize, batchSize):
    """
    Loads dataset.
    datName must be "mnist", "fashion_mnist", "cifar10", or "ASIRRA".
    """
    # All data should be here...
    # Build preprocessing classes
    normalize  = scaleTo01()
    patchitize = get_patches(patchSize)
#    vectorize  = vectorizeIm()
    vectorize  = transforms.Lambda(lambda x: x.view(-1, x.shape[-1]*x.shape[-2]))

##################################################################
    # Normalize,separate into patches
    m    = 500
    # nppi = math.floor(m/patchSize)**2
    # bsz  = fixBsz(batchSize,nppi )
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize((m,m))   # not sure why it's 28x28 in pytorch
                    # normalize,
                    # patchitize,
                    # vectorize
                    ])
#    # TRAINING SET
    transformed_images=[]
    for img in arrayData:
        # print(type(img))
        # print(img)
        trainset=transform(img)
        # trainset=torch.from_numpy(trainset).long()
        transformed_images.append(trainset)
    trainloader = torch.utils.data.DataLoader(transformed_images, batch_size=10,
                                                shuffle=False)
##################################################################

##################################################################
    return trainloader
##################################################################
## LOAD 2 DATASETS AND MIX (uses above classes)
##################################################################
#def loadMixData(datName1,datName2, patchSize, batchSize):#
#
#trDat1 = torchvision.datasets.ImageFolder(traindir, transform=transform)
#trDat2 = torchvision.datasets.ImageFolder(traindir, transform=transform)











