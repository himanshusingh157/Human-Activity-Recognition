from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
from sklearn.metrics import accuracy_score
import torch.nn as nn
from PIL import Image
import pickle
import os
import time
import matplotlib.pyplot as plt


class Dataset_3DCNN(data.Dataset):
    def __init__(self, data_path, folders, labels, frames, transform=None):
        self.data_path = data_path
        self.labels = labels
        self.folders = folders
        self.transform = transform
        self.frames = frames

    def __len__(self):
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        X = []
        for i in self.frames:
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i))).convert('L')
            if use_transform is not None:
                image = use_transform(image)
            X.append(image.squeeze_(0))
        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        folder = self.folders[index]
        X = self.read_images(self.data_path, folder, self.transform).unsqueeze_(0)
        y = torch.LongTensor([self.labels[index]]) 
        return X, y


class CNN3D(nn.Module):
    def __init__(self,t_dim=100,img_x=224,img_y=224,drop_p=0.2):
        super(CNN3D, self).__init__()
        self.t_dim=t_dim
        self.img_x=img_x
        self.img_y=img_y
        self.drop_p=drop_p

        self.conv1=nn.Conv3d(in_channels=1,out_channels=32,kernel_size=(5,5,5),stride=(2,2,2),padding=(0,0,0))
        self.bn1=nn.BatchNorm3d(32)
        self.conv2=nn.Conv3d(in_channels=32,out_channels=64,kernel_size=(3,3,3),stride=(2,2,2),padding=(0,0,0))
        self.bn2=nn.BatchNorm3d(64)
        self.relu=nn.ReLU(inplace=True)
        self.drop=nn.Dropout3d(self.drop_p)
        self.pool=nn.MaxPool3d(2)
        self.fc1=nn.Linear(64*5*62*84,256)
        self.fc2=nn.Linear(256,256)
        self.fc3=nn.Linear(256,101)

    def forward(self, x_3d):
        x=self.conv1(x_3d)
        x=self.bn1(x)
        x=self.relu(x)
        x=self.drop(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.relu(x)
        x=self.drop(x)
        x=x.view(x.size(0),-1)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=F.dropout(x,p=self.drop_p,training=self.training)
        x=self.fc3(x)
        return x

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss =0
    correct=0
    N_count = 0
    for X, y in train_loader:
        X, y =X.to(device), y.to(device).view(-1,)
        N_count+=X.size(0)
        optimizer.zero_grad()
        output=model(X)
        loss=F.cross_entropy(output, y,reduction='sum')
        total_loss+=loss.item()
        y_pred=torch.max(output, 1)[1]
        score=accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy(),normalize=False)
        correct+=score
        loss.backward()
        optimizer.step()
    return total_loss, correct/N_count


def test(model, device, optimizer, test_loader):
    model.eval()
    test_loss = 0
    all_y=[]
    all_y_pred=[]
    N_count=0
    with torch.no_grad():
        for X,y in test_loader:
            X,y=X.to(device), y.to(device).view(-1, )
            N_count+=X.size(0)
            output=model(X)
            loss=F.cross_entropy(output, y, reduction='sum')
            test_loss+=loss.item()
            y_pred=output.max(1, keepdim=True)[1]
            all_y.extend(y)
            all_y_pred.extend(y_pred)
    all_y=torch.stack(all_y, dim=0)
    all_y_pred=torch.stack(all_y_pred, dim=0)
    score=accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy(),normalize=False)
    return test_loss,score/N_count

data_path="./jpegs_256/" 
action_name_path="./UCF101actions.pkl"
dropout=0.1
device=torch.device("cuda")
img_x, img_y=256, 342

with open(action_name_path, 'rb') as f:
    action_names=pickle.load(f)

le=LabelEncoder()
le.fit(action_names)
action_category=le.transform(action_names).reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(action_category)

fnames=os.listdir(data_path)
actions=[]
all_X_list=[]
for f in fnames:
    loc1=f.find('v_')
    loc2=f.find('_g')
    actions.append(f[(loc1 + 2): loc2])
    all_X_list.append(f)

all_y_list=le.transform(actions)

train_list,test_list,train_label,test_label=train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=99)
transform=transforms.Compose([transforms.Resize([img_x, img_y]),transforms.ToTensor(),transforms.Normalize(mean=[0.5], std=[0.5])])
selected_frames=np.arange(1,29).tolist()

train_set=Dataset_3DCNN(data_path, train_list, train_label, selected_frames, transform=transform)
test_set=Dataset_3DCNN(data_path, test_list, test_label, selected_frames, transform=transform)

train_loader=data.DataLoader(train_set,batch_size=16,shuffle=True,num_workers=4,pin_memory=True)
test_loader=data.DataLoader(test_set,batch_size=16,shuffle=True,num_workers=4,pin_memory=True)

model=CNN3D(t_dim=28,img_x=img_x,img_y=img_y,drop_p=dropout).to(device)

if torch.cuda.device_count() > 1:
    model=nn.DataParallel(model)

optimizer=torch.optim.Adam(model.parameters(), lr=1e-5)

train_losses=[]
train_scores=[]
test_losses=[]
test_scores=[]

best_acc=0

for epoch in range(1,51):
    start=time.time()
    print(f"Epoch = {epoch}")
    train_loss, train_score = train(model, device, train_loader, optimizer, epoch)
    print(f"Training loss = {train_loss}   Training accuracy = {train_score}")
    test_loss, test_score = test(model, device, optimizer, test_loader)
    print(f"Test loss = {test_loss}   Test accuracy = {test_score}")
    end=time.time()
    print(f"Time Elapsed = {end - start}")

    if best_acc<test_score:
        best_acc=test_score
        torch.save(model.state_dict(), 'Final.pt')
    
    if epoch%5==0:
        filename=f"./models/{epoch}.pt"
        torch.save(model.state_dict(), filename)

    train_losses.append(train_loss)
    train_scores.append(train_score*100)
    test_losses.append(test_loss)
    test_scores.append(test_score*100)


plt.plot(np.arange(1,21), train_losses[:20])  # train loss (on epoch end)
plt.plot(np.arange(1, 21), test_losses[:20])         #  test loss (on epoch end)
plt.title("Model Loss")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['train', 'test'])
plt.savefig("Loss.png", dpi=600)
plt.show()

plt.plot(np.arange(1, 21), train_scores[:20])  # train accuracy (on epoch end)
plt.plot(np.arange(1, 21), test_scores[:20])         #  test accuracy (on epoch end)
plt.title("Model Accuracy")
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'])
plt.savefig("Accu.png", dpi=600)
plt.show()
