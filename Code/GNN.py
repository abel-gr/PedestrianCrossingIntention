from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric_temporal.nn import GConvGRU

from torch import nn
from torch import flatten
from torch import reshape

import math

#from GraphConvolution import *


class SpatialTemporalGNN(nn.Module):
    
    def __init__(self, embed_dim, outputSize, numNodes, net='GConvGRU', filterSize=3):
        
        super(SpatialTemporalGNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.outputSize = outputSize
        self.numNodes = numNodes
                
        # Definition of Conv layers:
        
        conv1mult = 1
        
        size_in0 = embed_dim
        size_out0 = size_in0 * conv1mult
            
        if net == 'GConvGRU':
            
            self.conv1 = GConvGRU(size_in0, size_out0, filterSize)
            
            self.conv2 = GConvGRU(size_out0, size_out0, filterSize)
            
            
        
        # Definition of linear layers:
                
        self.size_in1 = size_out0 * numNodes
        size_out1 = int(self.size_in1 * 0.5)
        self.lin1 = nn.Linear(self.size_in1, size_out1)
        
        size_in2 = size_out1
        size_out2 = int(size_in2 * 0.5)
        self.lin2 = nn.Linear(size_in2, size_out2)
        
        self.lin3 = nn.Linear(size_out1, outputSize)
        
        
        # Definition of extras
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.7)
        
        # Definition of activation functions
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    
    def forward(self, data):
        
        x_list, edge_index, edge_weight, batch = data.x_temporal, data.edge_index, data.edge_weight, data.batch
        
        H_i = None
        
        # x_list is a list with the node features of each temporal moment
        # for each temporal moment get the corresponding node features:
        for x_i in x_list:
            
            # x: Node features
            # edge_index: Edges connectivity in COO format
            # edge_weight: Weights of the edges
            # H: Hidden state matrix for all nodes
            H_i = self.conv1(X=x_i, edge_index=edge_index, edge_weight=edge_weight, H=H_i)
            
            H_i = self.dropout5(H_i)
            
            H_i = self.relu(H_i)
            
        x = H_i
                        
        #x = self.relu(x)

        x = x.view(int(math.ceil(batch.shape[0]/self.numNodes)), self.size_in1)
        
        x = self.lin1(x)
        
        x = self.dropout5(x)

        x = self.relu(x)
        
        #x = self.lin2(x)
        
        #x = self.dropout5(x)

        #x = self.relu(x)

        x = self.lin3(x)
        
        x = self.softmax(x)
        
        return x


    
class SpatialGNN(nn.Module):
    
    def __init__(self, embed_dim, outputSize, numNodes, pool_ratio=None):
        
        super(SpatialGNN, self).__init__()

        self.embed_dim = embed_dim
        self.outputSize = outputSize
        self.numNodes = numNodes
        self.pool_ratio = pool_ratio
        
        conv1mult = 1
        lin1mult = 1
        
        size_in0 = embed_dim
        size_out0 = size_in0 * conv1mult
        self.conv1 = GCNConv(size_in0, size_out0)
        
        if pool_ratio is not None:
            pool_ratio1 = 0.8
            self.pool1 = TopKPooling(size_out0, ratio=pool_ratio1)
                
            self.nodes2 = int(numNodes*pool_ratio1)
        else:
            self.nodes2 = numNodes
        
        self.size_in1 = size_out0 * self.nodes2
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
        
        #x = self.dropout(x)
                
        x = self.relu(x)
        
        if self.pool_ratio is not None:
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = x.view(int(math.ceil(batch.shape[0]/self.nodes2)), self.size_in1)
        
        x = self.lin1(x)
        
        x = self.dropout(x)

        x = self.relu(x)

        x = self.lin2(x)
        
        x = self.softmax(x)
        
        return x