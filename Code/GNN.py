import torch

from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool, global_max_pool

from torch_geometric_temporal.nn import GConvGRU, GConvLSTM, TGCN, DCRNN, GCLSTM

from torch import nn
from torch import flatten
from torch import reshape
from torch import matmul
from torch import zeros

import torch.nn.functional as F

import math

class SpatialTemporalGNN(nn.Module):
    
    def __init__(self, embed_dim, outputSize, numNodes, net='GConvGRU', filterSize=3, dropout=0.3, batchSize=1000, compute_explainability=False, compute_attention=False):
        
        super(SpatialTemporalGNN, self).__init__()
        
        self.embed_dim = embed_dim
        self.outputSize = outputSize
        self.numNodes = numNodes
        self.convNet = net
        self.compute_explainability = compute_explainability
        self.compute_attention = compute_attention
        
        if self.compute_explainability:
            self.last_conv_act = None
        
        if self.compute_attention:
            self.attended_h = [] #zeros((numNodes, filterSize))
                
        # Definition of Conv layers:
        
        conv1mult = 1
        
        size_in0 = embed_dim
        size_out0 = size_in0 * conv1mult
            
        if net == 'GConvGRU':
            
            self.conv1 = GConvGRU(size_in0, size_out0, filterSize)
            
            self.conv2 = GConvGRU(size_out0, size_out0, filterSize)
            
        elif net == 'GConvLSTM':
            
            self.conv1 = GConvLSTM(size_in0, size_out0, filterSize)
            
            self.conv2 = GConvLSTM(size_out0, size_out0, filterSize)
            
            
        elif net == 'TGCN':
            
            self.conv1 = TGCN(size_in0, size_out0)
            
            self.conv2 = TGCN(size_out0, size_out0)
            
        elif net == 'DCRNN':
            
            self.conv1 = DCRNN(size_in0, size_out0, filterSize)
            
            self.conv2 = DCRNN(size_out0, size_out0, filterSize)
            
        elif net == 'GCLSTM':
            
            self.conv1 = GCLSTM(size_in0, size_out0, filterSize)
            
            self.conv2 = GCLSTM(size_out0, size_out0, filterSize)
            
        
        # Definition of linear layers:
                
        self.size_in1 = size_out0 * numNodes
        size_out1 = int(self.size_in1 * 0.5)
        self.lin1 = nn.Linear(self.size_in1, size_out1)
        
        size_in2 = size_out1
        size_out2 = int(size_in2 * 0.5)
        self.lin2 = nn.Linear(size_in2, size_out2)
        
        self.lin3 = nn.Linear(size_out2, outputSize)
        
        
        # Definition of extras
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.dropout = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout7 = nn.Dropout(p=0.7)
        
        
        # Definition of activation functions
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    
    def set_grad(self, var):
        def grad_var(grad):
            var.grad = grad
        return grad_var
        
    
    def forward(self, data):
        
        x_list, edge_index, edge_weight, batch = data.x_temporal, data.edge_index, data.edge_weight, data.batch

        batch_size = int(data.batch[-1].detach().cpu()) + 1
        
        H_i = None
        
        # x_list is a list with the node features of each temporal moment
        # for each temporal moment get the corresponding node features:
        for x_i in x_list:
            
            # x: Node features
            # edge_index: Edges connectivity in COO format
            # edge_weight: Weights of the edges
            # H: Hidden state matrix for all nodes
            
            if self.compute_explainability:
                with torch.enable_grad():
                    H_i = self.conv1(X=x_i, edge_index=edge_index, edge_weight=edge_weight, H=H_i)

                    if self.convNet == 'GConvLSTM' or self.convNet == 'GCLSTM':
                        H_i = H_i[0]
                                                
            else:
                H_i = self.conv1(X=x_i, edge_index=edge_index, edge_weight=edge_weight, H=H_i)

                if self.convNet == 'GConvLSTM' or self.convNet == 'GCLSTM':
                    H_i = H_i[0]
                    
            if self.compute_attention:
                similarity_matrix = matmul(x_i, H_i.t())
                attention_weights = F.softmax(similarity_matrix, dim=-1)
                attended_hidden_state = matmul(attention_weights, H_i)

                for attended_i, _ in enumerate(attended_hidden_state[0::self.numNodes]): # Because of using batch
                    self.attended_h.append(attended_hidden_state[attended_i:attended_i+self.numNodes].detach().cpu().numpy())
            
            
            if self.compute_explainability:
                self.last_conv_act = H_i
                self.last_conv_act.register_hook(self.set_grad(self.last_conv_act))
            
            H_i = self.dropout(H_i)
            
            H_i = self.relu(H_i)
            
        x = H_i
                        
        #x = self.relu(x)

        x = x.view(int(math.ceil(batch.shape[0]/self.numNodes)), self.size_in1)
        
        x = self.lin1(x)
        
        x = self.dropout(x)

        x = self.relu(x)
        
        x = self.lin2(x)
        
        x = self.dropout(x)

        x = self.relu(x)

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