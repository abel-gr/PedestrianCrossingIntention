from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch import nn
from torch import flatten
from torch import reshape

import math

#from GraphConvolution import *

class SpatialGNN(nn.Module):
    
    def __init__(self, embed_dim, outputSize, numNodes):
        
        super(SpatialGNN, self).__init__()

        self.embed_dim = embed_dim
        self.outputSize = outputSize
        self.numNodes = numNodes
        
        conv1mult = 2
        lin1mult = 2
        
        size_in0 = embed_dim
        size_out0 = size_in0 * conv1mult
        self.conv1 = GCNConv(size_in0, size_out0)
        
        self.size_in1 = size_out0 * numNodes
        size_out1 = self.size_in1 * lin1mult
        self.lin1 = nn.Linear(self.size_in1, size_out1)
        
        size_in2 = size_out1
        size_out2 = outputSize
        self.lin2 = nn.Linear(size_in2, outputSize)
        
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=0.3)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        
    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        
        x = self.dropout(x)
                
        x = self.relu(x)

        x = reshape(x, (int(math.ceil(batch.shape[0]/self.numNodes)), self.size_in1))
        
        x = self.lin1(x)
        
        x = self.dropout(x)

        x = self.relu(x)

        x = self.lin2(x)
        
        x = self.softmax(x)
        
        return x