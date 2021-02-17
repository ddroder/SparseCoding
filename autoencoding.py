import os 
import glob
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from tensorflow.keras import layers,losses
from tensorflow.keras.datasets import mnist
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
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter


class AE(nn.Module):
    #this class will create an autoencoding object
    #the objective is to simply create the autoencoder object
    #and then pass in training data to train the model
    #to very simply create the autoencoder
    def __init__(self, **kwargs):
        super().__init__()
        model_dir="/nvme_ssd/bensCode/SparseCoding/models/model3.pt"
        self.encoder=nn.Sequential(torch.load(model_dir)
        )
        self.encoder.eval()
        self.encoder2=nn.Sequential(
            nn.Linear(in_features=1024,out_features=121)
            )
        self.encoder2.eval()
        self.decoder = nn.Sequential(
            nn.Linear(in_features=121, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=1024),
            nn.Sigmoid()
        )
    def trainAE(self,epochs,train_data,save=False,decoderName=1,writeTb=True):
        """
        this is a method that will train the autoencoder.
        After this runs, it will update the constructed AE (auto encoder)
        to be a trained model.
        """
        for epoch in range(epochs):
            loss = 0
            for batch_features, _ in train_data:
                batch_features = batch_features.view(-1, 1024).to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_features)
                
                train_loss = criterion(outputs, batch_features)
                
                train_loss.backward()
                
                optimizer.step()
                
                loss += train_loss.item()
            
                

            loss = loss / len(train_data)
            if writeTb:
                tb.add_scalar("Loss",loss,epoch)
                tb.close()
            
            print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))
        if save:
            self.saveModel(modelName=decoderName)
    def forward(self, features):
        """
        this method is the logic for the forward prop. of the model.
        """
        x=self.encoder(features)
        x=self.encoder2(x)
        x=self.decoder(x)
        return x
    def getEncoderImage(self,x):
        """
        this method will pass an image through the encoder
        and returns that image. This is mostly just for
        adding the image to the tensorboard display.
        """
        encoded_image=self.encoder(x)
        encoded_image=self.encoder2(encoded_image)
        return encoded_image
    def saveModel(self,modelName=1):
        """
        this method will contain the code for saving the decoder model
        """
        model_dir="/nvme_ssd/bensCode/SparseCoding/models/Decoder_1.pt"
        torch.save(self.decoder,model_dir)
        print(f"model saved to {model_dir}")




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tb=SummaryWriter()
model = AE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

criterion = nn.MSELoss()

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


trainload,testload=loadData("mnist",32,10)
images,labels=next(iter(trainload))
grid=torchvision.utils.make_grid(images)
images=images.to('cuda:0')
tb.add_graph(model,images)
model.trainAE(epochs=10,train_data=trainload,save=True)
train_data=[]
for batch_idx, (true_sigs, im_labels) in enumerate(testload):
    train_data.append(true_sigs)

im_test=train_data[12][0]
im_test2=im_test.reshape(32,32)
im_test2=im_test.to('cuda:0')
pred=model(im_test2)
pred1=pred.cpu().data.numpy()
pred1=pred1.reshape(32,32)
pred=pred.cpu().reshape(32,32)
img_grid=torchvision.utils.make_grid(pred)
tb.add_image("Prediction",img_grid)

fig,axs=plt.subplots(1,3,sharey='row')
im_test2=im_test2.reshape(32,32)
im_test=im_test.to('cuda:0')
encoded_img=model.getEncoderImage(x=im_test)
encoded_img_tb=encoded_img.cpu().reshape(11,11)
img_grid=torchvision.utils.make_grid(encoded_img_tb)
tb.add_image("Encoded",img_grid)
encoded_img=encoded_img.cpu().data.numpy()
encoded_img=encoded_img.reshape(11,11)
im_test2=im_test2.cpu()
im_test3=im_test2.reshape(32,32)
img_grid=torchvision.utils.make_grid(im_test3)
tb.add_image("Original",img_grid)
tb.close()
axs[0].set_title("Original Image")
axs[0].imshow(im_test2)
print(f"shape of og_img:{im_test2.shape}")
axs[1].set_title("Encoded Image")
axs[1].imshow(encoded_img)
print(f"shape of encoded_img:{encoded_img.shape}")
axs[2].set_title("Decoded Image")
axs[2].imshow(pred1)
print(f"shape of decoded_img:{pred1.shape}")
plt.subplots_adjust(wspace=.8)
plt.show()




