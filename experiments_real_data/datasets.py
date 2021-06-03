
import numpy as np
import pandas as pd

import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
        
def GetDataset(name, base_path):
    """ Load a dataset

    Parameters
    ----------
    name : string, dataset name
    base_path : string, e.g. "path/to/datasets/directory/"

    Returns
    -------
    X : features (nXp)
    y : labels (n)

	"""
    
    if name=='mnist':
            
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(base_path, train=True, download=True,
                           transform=transforms.ToTensor()),batch_size=1, shuffle=False)
        
        X = np.zeros((28*28,len(train_loader)),dtype=np.float32)
        y = np.zeros(len(train_loader),dtype=np.int)
        for i, (data, target) in enumerate(train_loader):
            X[:,i] = data.view(-1)
            y[i] = target
            
        y = y - min(y)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X.T)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        
    if name=='svhn':
            
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(base_path, download=True,
                           transform=transforms.ToTensor()),batch_size=1, shuffle=False)
        
        X = np.zeros((32*32*3,len(train_loader)),dtype=np.float32)
        y = np.zeros(len(train_loader),dtype=np.int)
        for i, (data, target) in enumerate(train_loader):
            X[:,i] = data.view(-1)
            y[i] = target
        
        y = y - min(y)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X.T)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    if name=='fashion':
            
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(base_path + 'fashion/', train=True, download=True,
                           transform=transforms.ToTensor()),batch_size=1, shuffle=False)
        
        X = np.zeros((28*28,len(train_loader)),dtype=np.float32)
        y = np.zeros(len(train_loader),dtype=np.int)
        for i, (data, target) in enumerate(train_loader):
            X[:,i] = data.view(-1)
            y[i] = target
        
        y = y - min(y)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X.T)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if name=='cifar10':
            
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(base_path, train=True, download=True,
                       transform=transforms.ToTensor()),batch_size=1, shuffle=False)
    
        X = np.zeros((32*32*3,len(train_loader)),dtype=np.float32)
        y = np.zeros(len(train_loader),dtype=np.int)
        for i, (data, target) in enumerate(train_loader):
            X[:,i] = data.view(-1)
            y[i] = target
            
        y = y - min(y)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X.T)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    if name=='cifar10lt':
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(base_path, train=True, download=True,
                       transform=transforms.ToTensor()),batch_size=1, shuffle=False)

        X = np.zeros((32*32*3,len(train_loader)),dtype=np.float32)
        y = np.zeros(len(train_loader),dtype=np.int)
        for i, (data, target) in enumerate(train_loader):
            X[:,i] = data.view(-1)
            y[i] = target
            
        y = y - min(y)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X.T)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        idxs_keep = np.argwhere(y != 0)[:, 0]
        idxs_remove = np.argwhere(y == 0)[:, 0]
        idxs_save = np.random.choice(idxs_remove, int(len(idxs_remove) * 0.2))
        idxs = np.concatenate([idxs_keep, idxs_save])

        X = X[idxs]
        y = y[idxs]

    if name=='cifar100':
        train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR100(base_path, train=True, download=True,
                       transform=transforms.ToTensor()),batch_size=1, shuffle=False)
    
        X = np.zeros((32*32*3,len(train_loader)),dtype=np.float32)
        y = np.zeros(len(train_loader),dtype=np.int)
        for i, (data, target) in enumerate(train_loader):
            X[:,i] = data.view(-1)
            y[i] = target
        
        y = y - min(y)
        pca = PCA(n_components=50)
        X = pca.fit_transform(X.T)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    if name=='ecoli':
        
        
        data=pd.read_csv(base_path + 'ecoli.csv')
    
        X = data.iloc[:,1:-1]
        X = pd.get_dummies(X).values
        y = data.iloc[:,-1]
        y = y.astype('category').cat.codes.values
        y = y - min(y)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    if name =='mice':
        
        try:
            import xlrd
        except:
            raise ImportError("To load this dataset, you need the library 'xlrd'. Try installing: pip install xlrd")
        url="https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls"
        data=pd.read_excel(url, header=0, na_values=['', ' '])
        features = data.iloc[:,1:-4]
        features = features.fillna(value=0)
        X = pd.get_dummies(features).values
        labels = data.iloc[:,-1]
        y = labels.astype('category').cat.codes.values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
    if name == 'imagenet':
        data_base_path = '/future/u/jtguibas/CCAPS'
        softmax_fname = 'imagenet_softmax_val.npy'
        labels_fname = 'imagenet_labels_val.npy'
        print("=> Loading softmax scores")
        X = np.load(os.path.join(data_base_path, softmax_fname))
        print('=> Loading labels')
        y = np.load(os.path.join(data_base_path, labels_fname))
        print('=> ImageNet load successful')

    X = X.astype(np.float32)
    y = y.astype(np.float32)

    return X, y
