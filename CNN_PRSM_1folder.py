import os
import cv2
import torch
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils
import torch.utils.data

import os
import shutil
import random

# def move_random_images(source_folder, destination_folder, percentage=0.2):
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     all_images = [f for f in os.listdir(source_folder)]

#     num_images_to_move = int(len(all_images) * percentage)

#     selected_images = random.sample(all_images, num_images_to_move)

#     for image in selected_images:
#         source_path = os.path.join(source_folder, image)
#         destination_path = os.path.join(destination_folder, image)
#         shutil.move(source_path, destination_path)

#     print(f"moved {num_images_to_move} images to {destination_folder}")

# source_folder =  r'/home/mohsen/Downloads/PRSM2/Train/10'
# destination_folder =  r'/home/mohsen/Downloads/PRSM2/Test/10' 

# move_random_images(source_folder, destination_folder)


Train_path = r'/home/mohsen/Downloads/PRSM2/Train'
Test_path = r'/home/mohsen/Downloads/PRSM2/Test'
transform = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()
                               ,transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)) ])
Train_dataset = torchvision.datasets.ImageFolder(root=Train_path,transform=transform)
Test_dataset = torchvision.datasets.ImageFolder(root=Test_path,transform=transform)
Train_loader = torch.utils.data.DataLoader(dataset=Train_dataset,batch_size=4,shuffle=True)
Test_loader = torch.utils.data.DataLoader(dataset=Test_dataset,batch_size=4,shuffle=True)
# print(len(Test_loader))

# a = iter(Test_loader)
# image,label = next(a)
# print(len(image))
# print(len(label))


class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=(3,3),padding=1)
        self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2))
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2))
        self.fc1 = nn.Linear(8*8*32,10)


    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        return x



model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01)
num_epochs = 10

for epoch in range(num_epochs):
    for i,(images,labels) in enumerate(Train_loader):
        images = images
        labels = labels
        output = model(images)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            print(f'epoch:{epoch+1}/{num_epochs} and step: {(i+1)}/{(len(Train_loader))}, loss: {loss.item():0.5f}')






classes = [1,2,3,4,5,6,7,8,9,10]
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in Test_loader:
        images = images
        labels = labels
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(4):
            try:
                label = labels[i]                
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
            except(IndexError):
                continue
    
        


    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(10):
        acc = 100.0 * n_class_correct[i] / n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc} %')









