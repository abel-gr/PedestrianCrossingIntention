from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch import nn
from torch import flatten
from torch import reshape

import math

#from GraphConvolution import *

class GNN(nn.Module):
    
    def __init__(self, embed_dim, outputSize, numNodes):
        
        super(GNN, self).__init__()

        self.embed_dim = embed_dim
        self.outputSize = outputSize
        self.numNodes = numNodes        
        
        self.conv1 = GCNConv(embed_dim, embed_dim)
        self.lin1 = nn.Linear(numNodes * embed_dim, numNodes * embed_dim)
        self.lin2 = nn.Linear(numNodes * embed_dim, outputSize)
        self.softmax = nn.Softmax(dim=-1)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)

        x = self.relu(x)

        x = reshape(x, (int(math.ceil(batch.shape[0]/self.numNodes)), self.numNodes * self.embed_dim))
        
        x = self.lin1(x)

        x = self.relu(x)

        x = self.lin2(x)

        x = self.softmax(x)

        return x