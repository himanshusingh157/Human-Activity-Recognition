import shutil
import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils import data
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np


#list1 contain all the classes and number of images in particular class
#train_list contains the name of all training images
#test_list contai the name of all test images

with open('./dataset/ImageSplits/actions.txt') as f:
    contents = f.readlines()
list1=[]
for content in contents[1:]:
    temp1=content.split('\t')
    temp2=[]
    for i in temp1:
        if i!='':
            temp2.append(i)
    temp2[-1]=temp2[-1][:-1]
    list1.append(temp2)

src_dir = "./dataset/JPEGImages"
for temp1 in list1:
    n=int(temp1[1])
    len_train=int(n*0.7)
    for i in range(1,len_train):
        image_name=temp1[0]+'_'+str(i).zfill(3)+'.jpg'
        dst_dir = "dataset/train/"+temp1[0]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(os.path.join(src_dir, image_name), dst_dir)
        
    for i in range(len_train,n+1):
        image_name=temp1[0]+'_'+str(i).zfill(3)+'.jpg'
        dst_dir = "dataset/test/"+temp1[0]
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(os.path.join(src_dir, image_name), dst_dir)


def train(model,train_loader,loss_fn,optimizer):
    correct=0
    total_loss=0
    total_images=0
    for images,labels in train_loader:
        images=images.to(device)
        labels=labels.to(device)
        output=model(images)
        loss=loss_fn(output,labels,reduction='sum')
        _,predicted=torch.max(output,dim=1)
        correct+=(predicted==labels).sum().item()
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        total_images+=len(images)
        optimizer.zero_grad()
    return total_loss,correct/total_images

def test(model,test_loader,loss_fn):
    correct=0
    total_loss=0
    total_images=0
    with torch.no_grad():
        for images,labels in test_loader:
            images=images.to(device)
            labels=labels.to(device)
            output=model(images)
            loss=loss_fn(output,labels,reduction='sum')
            _,predicted=torch.max(output,dim=1)
            correct+=(predicted==labels).sum().item()
            total_loss+=loss.item()
            total_images+=len(images)
    return total_loss,correct/total_images


def set_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr

img_size=224
rescale=int(img_size*1.25)
batch_size=64
learning_rate=0.0001


device = torch.device("cuda")
train_path="dataset/train/"
test_path="dataset/test"

transformations=transforms.Compose([transforms.Resize((rescale, rescale)),transforms.RandomCrop((img_size,img_size)),transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.ToTensor()])
train_data=torchvision.datasets.ImageFolder(train_path, transform=transformations)
train_loader=torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True )

test_transform=transforms.Compose([transforms.Resize((img_size,img_size)),transforms.ToTensor()])
test_data=torchvision.datasets.ImageFolder(test_path, transform=test_transform)
test_loader=torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=4, shuffle=True)


model_name= "DenseNet-121"
print("Please enter Corresponding no. for the Model Name on which you want to train. The avialable options are :")
print("1\t\tVGG-16 \n2\t\tResNet-50\n3\t\tMobileNet\n4\t\tDensenet\nDefault\t\tDenseNet")
try:
    temp=int(input())
except ValueError:
    pass
if temp==1:
    model_name="VGG-16"
elif temp==2:
    model_name="ResNet-50"
elif temp==3:
    model_name="MobileNet-V2"
elif temp==4:
    model_name="DenseNet-121"
else:
    print("Wrong Input; Going to defalut choice")
print(f"Model chosen: {model_name}")


ckpt_dir = os.path.join("./models/", "checkpoints")
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)


if model_name == "VGG-16":
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, 40, True)
elif model_name == "ResNet-50":
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 40, True)
elif model_name == "MobileNet-V2":
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[-1] = nn.Linear(1280, 40, True)    
elif model_name == "DenseNet-121":
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, 40, True)


model.to(device)
loss_fn = F.cross_entropy


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


model.train()
train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]
optimizer.zero_grad()
best_acc=0

lr_list=[0.00005,0.00001,0.000005,0.000001]
lr_i=0

for epoch in range(1,501):
    print(f"Epoch {epoch}")
    loss,acc=train(model,train_loader,loss_fn,optimizer)
    train_loss.append(loss)
    train_acc.append(acc*100)
    print(f"Training Loss = {loss}    Training Accuracy ={acc*100}%")
    
    loss,acc=test(model,test_loader,loss_fn)
    test_loss.append(loss)
    test_acc.append(acc*100)
    print(f"Test Loss = {loss}    Test Accuracy ={acc*100}%")
    
    if best_acc<acc:
        best_acc=acc
        filename=ckpt_dir+f"/Final_{model_name}.pt"
        torch.save(model.state_dict(), filename)
    
    if epoch%25==0:
        filename=ckpt_dir+f"/{epoch}.pt"
        torch.save(model.state_dict(), filename)
        
    if epoch%100==0:
        set_lr(optimizer, lr_list[lr_i])
        lr_i+=1


plt.plot(np.arange(1,len(train_loss)+1),train_loss,label='Training Loss')
plt.plot(np.arange(1,len(test_loss)+1),test_loss,label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
filename=f"loss_{model_name}.png"
plt.savefig(filename)
plt.show()


plt.plot(np.arange(1,len(train_acc)+1),train_acc,label='Training Accuracy')
plt.plot(np.arange(1,len(test_acc)+1),test_acc,label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy')
plt.legend()
filename=f"Acc_{model_name}.png"
plt.savefig(filename)
plt.show()
