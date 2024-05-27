import torch 
import numpy as np
from torch.utils.data import TensorDataset,Dataset,DataLoader
from torch.utils.data import RandomSampler,BatchSampler 

train_data = np.load("ESM_feat_lable.npy")
lables = torch.tensor(train_data[:, 0], dtype=torch.float32)
features = torch.tensor(train_data[:, 1:], dtype=torch.float32)
ds = TensorDataset(lables,features)
dl = DataLoader(ds,batch_size=16,drop_last = False)
lables,features = next(iter(dl))
print("features = ",features,features.shape)
print("labels = ",lables,lables.shape)