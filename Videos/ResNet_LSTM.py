from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import accuracy_score
import pickle
import torchvision.transforms as transforms
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch
import os
import torchvision.models as models
from PIL import Image
import torch.nn.functional as F
import time

class Dataset_CRNN(data.Dataset):
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
            image = Image.open(os.path.join(path, selected_folder, 'frame{:06d}.jpg'.format(i)))
            if use_transform is not None:
                image = use_transform(image)
            X.append(image)
        X = torch.stack(X, dim=0)
        return X

    def __getitem__(self, index):
        folder = self.folders[index]
        X = self.read_images(self.data_path, folder, self.transform)
        y = torch.LongTensor([self.labels[index]])
        return X, y


class ResCNNEncoder(nn.Module):
    def __init__(self, drop_p=0.3):
        super(ResCNNEncoder, self).__init__()
        self.drop_p = drop_p
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1] 
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, 1024)
        self.bn1 = nn.BatchNorm1d(1024, momentum=0.01)
        self.fc2 = nn.Linear(1024, 786)
        self.bn2 = nn.BatchNorm1d(786, momentum=0.01)
        self.fc3 = nn.Linear(786,512)
        
    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t, :, :, :])
                x = x.view(x.size(0), -1)
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)
            cnn_embed_seq.append(x)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        return cnn_embed_seq


class DecoderRNN(nn.Module):
    def __init__(self,drop_p=0.1):
        super(DecoderRNN, self).__init__()
        self.drop_p = drop_p
        self.LSTM = nn.LSTM(input_size=512,hidden_size=512,num_layers=3,batch_first=True)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256,101)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        x = self.fc1(RNN_out[:, -1, :])
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x

def train(model,train_loader, optimizer):
    cnn, rnn=model
    cnn.train()
    rnn.train()

    losses=0
    scores=0
    N_count = 0
    for X, y in train_loader:
        X, y = X.to(device),y.to(device).view(-1)
        N_count += X.size(0)
        optimizer.zero_grad()
        output = rnn(cnn(X))
        loss = F.cross_entropy(output, y,reduction='sum')
        losses+=loss.item()
        y_pred = torch.max(output, 1)[1]
        step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy(),normalize=False)
        scores+=step_score
        loss.backward()
        optimizer.step()
    return losses, scores/N_count

def test(model,optimizer, test_loader):
    cnn, rnn=model
    cnn.eval()
    rnn.eval()
    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device).view(-1)
            output = rnn(cnn(X))
            loss = F.cross_entropy(output, y, reduction='sum')
            test_loss += loss.item()
            y_pred = output.max(1, keepdim=True)[1]
            all_y.extend(y)
            all_y_pred.extend(y_pred)

    all_y = torch.stack(all_y, dim=0)
    all_y_pred = torch.stack(all_y_pred, dim=0)
    test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())
    return test_loss, test_score

data_path = "./jpegs_256/"
action_name_path = './UCF101actions.pkl'

dropout_p = 0.0
device = torch.device("cuda")

with open(action_name_path, 'rb') as f:
    action_names = pickle.load(f)

le=LabelEncoder()
le.fit(action_names)
action_category=le.transform(action_names).reshape(-1, 1)

enc=OneHotEncoder()
enc.fit(action_category)

actions=[]
fnames=os.listdir(data_path)

all_X_list = []
for f in fnames:
    loc1=f.find('v_')
    loc2=f.find('_g')
    actions.append(f[(loc1 + 2): loc2])
    all_X_list.append(f)
    
all_y_list=le.transform(actions)

train_list,test_list,train_label,test_label=train_test_split(all_X_list, all_y_list, test_size=0.25, random_state=42)

transform = transforms.Compose([transforms.Resize([224,224]),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

selected_frames = np.arange(1,29).tolist()

train_set=Dataset_CRNN(data_path, train_list, train_label, selected_frames, transform=transform)
train_loader=data.DataLoader(train_set,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)

test_set=Dataset_CRNN(data_path, test_list, test_label, selected_frames, transform=transform)
test_loader=data.DataLoader(test_set,batch_size=64,shuffle=True,num_workers=4,pin_memory=True)

cnn= ResCNNEncoder(drop_p=dropout_p).to(device)
rnn= DecoderRNN(drop_p=dropout_p).to(device)

if torch.cuda.device_count() > 1:
    cnn=nn.DataParallel(cnn,device_ids=[0,1])
    rnn=nn.DataParallel(rnn,device_ids=[0,1])
    model_params=list(cnn.module.fc1.parameters())+list(cnn.module.bn1.parameters())+list(cnn.module.fc2.parameters())+list(cnn.module.bn2.parameters())+list(cnn.module.fc3.parameters()) + list(rnn.parameters())

elif torch.cuda.device_count() == 1:
    model_params=list(cnn.fc1.parameters())+list(cnn.bn1.parameters())+list(cnn.fc2.parameters())+list(cnn.bn2.parameters())+list(cnn.fc3.parameters()) + list(rnn.parameters())

optimizer = torch.optim.Adam(model_params, lr=1e-4)

train_loss=[]
train_acc=[]
test_loss=[]
test_acc=[]

best_acc=0
for epoch in range(100):
    start=time.time()
    print(f"Epoch {epoch+1}")
    loss,acc=train([cnn, rnn],train_loader, optimizer)
    print(f"Training loss = {loss}\tTraining Accuracy ={acc*100}%")
    train_loss.append(loss)
    train_acc.append(acc*100)
    
    loss,acc=test([cnn, rnn],optimizer,test_loader)
    print(f"Test loss = {loss}\tTest Accuracy ={acc*100}%")
    test_loss.append(loss)
    test_acc.append(acc*100)
    end=time.time()
    print(f"Time Elapsed = {end-start}")
    
    if best_acc<test_score:
        best_acc=test_score
        torch.save(model.state_dict(), 'Final.pt')
    if epoch%5==0:
        filename=f"./models/{epoch}.pt"
        torch.save(model.state_dict(), filename)

plt.plot(np.arange(1,len(train_loss)+1),train_loss)
plt.plot(np.arange(1,len(test_loss)+1),test_loss)
plt.title("Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['train', 'test'], loc="upper left")
plt.savefig("loss.png", dpi=600)
plt.show()


plt.plot(np.arange(1,len(train_acc)+1),train_acc)
plt.plot(np.arange(1,len(test_acc)+1),test_acc)
plt.title("Training scores")
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['train', 'test'], loc="upper left")
plt.savefig("Acc.png", dpi=600)
plt.show()
