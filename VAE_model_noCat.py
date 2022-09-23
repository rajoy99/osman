from __future__ import print_function
import pandas as pd
from sklearn.model_selection import train_test_split
import collections
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
from torch.utils.data import TensorDataset,DataLoader


batch_size=128
epochs=5
seed=1
log_interval=10


torch.manual_seed(seed)



# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()

#Additional Info when using cuda
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


class VAE(nn.Module):
    def __init__(self,rows):
        super(VAE, self).__init__()
        self.rows=rows
        self.fc1 = nn.Linear(self.rows, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400,self.rows)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.rows))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar




def VAE_model(X_train,y_train,num_cols=[]): 


    #Formatting to handle categorical variables
    if not num_cols:
        num_cols=X_train.columns

    no_of_num_cols=len(num_cols)



    #Preprocessing 
    num_prep = make_pipeline(SimpleImputer(strategy='mean'),
                         MinMaxScaler())

    prep = ColumnTransformer([
        ('num', num_prep, num_cols)],
        remainder='drop')
    X_train_trans = prep.fit_transform(X_train)

    #minority determination
    minority_class=y_train.value_counts().index[-1]
    majority_class=y_train.value_counts().index[0]
    print("Minority and Majority class are: ",minority_class,majority_class)
    desired=y_train.value_counts().iloc[0] - y_train.value_counts().iloc[1]

    original_data_minority=[]
    original_data_majority=[]

    for i in range(len(y_train)):

        if y_train.iloc[i]==minority_class:
            original_data_minority.append(X_train_trans[i])
        else:
            original_data_majority.append(X_train_trans[i])


    original_data_majority=np.array(original_data_majority)
    original_data_minority=np.array(original_data_minority)


    data=original_data_minority
    rows=data.shape[1]
    print("Rows of the Org min",rows)

    data=torch.Tensor(data) 
    print("MIN,MAJ,Desired",original_data_minority.shape,original_data_majority.shape,desired)

    my_dataset = TensorDataset(data) 
    my_dataloader = DataLoader(my_dataset) 

    model = VAE(rows).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, rows), reduction='sum')


        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD




    def train(epoch):
        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(my_dataloader):
            data=data[0]
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(my_dataloader.dataset),
                    100. * batch_idx / len(my_dataloader),
                    loss.item() / len(data)))

        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(my_dataloader.dataset)))




    for epoch in range(1, epochs + 1):
        train(epoch)
        # with torch.no_grad():
        #     generated = torch.randn(desired, 20).to(device)
        #     generated = model.decode(generated).cpu()
        #     print(generated.shape)



    generated = torch.randn(desired, 20).to(device)
    generated = model.decode(generated).cpu()
    print("Peeking Type",type(generated.detach().numpy()))
    generated=generated.detach().numpy()
    totalX=np.vstack((generated,original_data_minority,original_data_majority))
    totalY=np.concatenate((np.full(original_data_majority.shape[0], minority_class),np.full(original_data_majority.shape[0],majority_class)),axis=None)

    print(totalX.shape,totalY.shape)

    for i in range(totalX.shape[0]):
        for j in range(no_of_num_cols,rows):
            totalX[i][j]=np.round(totalX[i][j])
        

    
    
    return totalX,totalY
    


