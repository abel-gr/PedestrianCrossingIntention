from torch import Tensor
from torch import nn
from torch import flatten
from torch import reshape

from torch_geometric.nn import GCNConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import knn_interpolate
from torch_geometric.nn import knn

import math
import numpy as np

def targeted_denoising(threshold=0.3):
    v = np.random.normal(loc=8, scale=3, size=10000)
    vi_unique_values, vi_counts = np.unique(v.astype(int), return_counts=True)
    
    for i in range(0, 26):
        if i not in vi_unique_values:
            vi_unique_values = np.append(vi_unique_values, i)
            vi_counts = np.append(vi_counts, 0)
            
    sorted_index = np.argsort(vi_unique_values)
    vi_unique_values = vi_unique_values[sorted_index]
    vi_counts = vi_counts[sorted_index]
    
    if np.min(vi_unique_values) < 0:
        zero_pos = np.argwhere(vi_unique_values == 0)[0][0]
        vi_unique_values = vi_unique_values[zero_pos:]
        vi_counts = vi_counts[zero_pos:]
        
    if vi_unique_values.shape[0] > 26:
        vi_unique_values = vi_unique_values[0:26]
        vi_counts = vi_counts[0:26]
                
    vi_counts_p = vi_counts / 10000
    
    n_th = vi_counts_p.reshape(-1, 1) + threshold
        
    return np.where(np.random.uniform(size=(vi_counts_p.shape[0], 1)) <= n_th, 0, 1)

class GraphConvolutionalAutoencoder(nn.Module):
    
    def __init__(self, embed_dim, numNodes):
        
        super(GraphConvolutionalAutoencoder, self).__init__()
        
        size_in1 = embed_dim
                
        size_out1 = math.floor(size_in1 * 2)
        self.conv1 = GCNConv(size_in1, size_out1)
        self.pool1 = TopKPooling(size_out1, ratio=0.5)
        
        size_out2 = math.floor(size_out1 * 2)
        self.conv2 = GCNConv(size_out1, size_out2)
        self.pool2 = TopKPooling(size_out2, ratio=0.5)
        
        size_out3 = math.floor(size_out2 * 2)
        self.conv3 = GCNConv(size_out2, size_out3)
        self.pool3 = TopKPooling(size_out3, ratio=0.5)
        
        size_out4 = math.floor(size_out3 * 0.5)
        self.conv4 = GCNConv(size_out3, size_out4)
        
        size_out5 = math.floor(size_out4 * 0.5)
        self.conv5 = GCNConv(size_out4, size_out5)
        
        size_out6 = math.floor(size_out5 * 0.5)
        self.conv6 = GCNConv(size_out5, size_out6)
        
        self.dropout5 = nn.Dropout(p=0.5)
        
        self.relu = nn.ReLU()
        
    
    def forward(self, data, noise='random'):
        
        x, edge_index, batch = data.x_temporal[0], data.edge_index, data.batch

        #print(x.shape)
        
        if noise == 'random':
            random_noise = Tensor(np.repeat(np.where(np.random.uniform(size=(self.numNodes, 1)) <= 0.5, 0, 1), self.embed_dim, axis=1))
            random_noise = random_noise.view(self.numNodes * self.embed_dim)
            x = x * random_noise
            
        elif noise == 'targeted':
            targeted_noise = Tensor(np.repeat(targeted_denoising(), self.embed_dim, axis=1))
            targeted_noise = targeted_noise.view(self.numNodes * self.embed_dim)
            x = x * targeted_noise
        
        x_prev_pool = []

        x_prev_pool.append(x)
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        #print(x.shape)
        x_prev_pool.append(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        #print(x.shape)
        x_prev_pool.append(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
        #print(x.shape)
        
        x = self.conv4(x, edge_index)
        x = self.relu(x)
        x = x[knn(x, x_prev_pool[-1], k=1)[1]]
        #print(x.shape)
        x = self.conv5(x, edge_index)
        x = self.relu(x)
        x = x[knn(x, x_prev_pool[-2], k=1)[1]]
        #print(x.shape)
        x = self.conv6(x, edge_index)
        x = self.relu(x)
        x = x[knn(x, x_prev_pool[-3], k=1)[1]]
        #print(x.shape)
        
        return x

class FC_Autoencoder(nn.Module):
    
    def __init__(self, embed_dim, numNodes):
        
        super(FC_Autoencoder, self).__init__()
        
        self.embed_dim = embed_dim
        self.numNodes = numNodes
        self.size_in1 = embed_dim * numNodes
        
        size_in1 = self.size_in1
                
        size_out1 = math.ceil(size_in1 * 0.8)
        self.lin1 = nn.Linear(size_in1, size_out1)
        
        size_out2 = math.ceil(size_out1 * 0.8)
        self.lin2 = nn.Linear(size_out1, size_out2)
        
        size_out3 = math.ceil(size_out2 * 0.8)
        self.lin3 = nn.Linear(size_out2, size_out3)
        
        size_out4 = math.ceil(size_out3 * 1.25)
        self.lin4 = nn.Linear(size_out3, size_out4)
        
        size_out5 = math.ceil(size_out4 * 1.25)
        self.lin5 = nn.Linear(size_out4, size_out5)
        
        self.lin6 = nn.Linear(size_out5, size_in1)
        
        self.relu = nn.ReLU()
        
        self.dropout5 = nn.Dropout(p=0.5)
        
    
    def forward(self, data, noise='random'):
        
        x, edge_index, batch = data.x_temporal[0], data.edge_index, data.batch
        #print(x.shape)
        
        x = x.view(int(math.ceil(x.shape[0]/self.numNodes)), self.size_in1)
        #print(x.shape)
        
        if noise == 'random':
            random_noise = Tensor(np.repeat(np.where(np.random.uniform(size=(self.numNodes, 1)) <= 0.5, 0, 1), self.embed_dim, axis=1))
            random_noise = random_noise.view(self.numNodes * self.embed_dim)
            x = x * random_noise
            
        elif noise == 'targeted':
            targeted_noise = Tensor(np.repeat(targeted_denoising(), self.embed_dim, axis=1))
            targeted_noise = targeted_noise.view(self.numNodes * self.embed_dim)
            x = x * targeted_noise

        x = self.lin1(x)
        x = self.relu(x)
        #print(x.shape)
        
        x = self.lin2(x)
        x = self.relu(x)
        #print(x.shape)
        
        x = self.lin3(x)
        x = self.relu(x)
        #print(x.shape)
        
        x = self.lin4(x)
        x = self.relu(x)
        #print(x.shape)
        
        x = self.lin5(x)
        x = self.relu(x)
        #print(x.shape)
        
        x = self.lin6(x)
        x = self.relu(x)
        #print(x.shape)
        
        return x