import torch
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing 
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
import math

class GraphLayer(MessagePassing): 
    def __init__(self, in_channels, out_channels,
                 negative_slope=0.2, dropout=0, bias=True, inter_dim=-1):
        

        super(GraphLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.negative_slope = negative_slope
        self.dropout = dropout


        self.lin = Linear(in_channels, out_channels, bias=False)

        self.a=Parameter(torch.Tensor(1, 4*out_channels))
        
        if bias:
            self.bias = Parameter(torch.Tensor( out_channels)) 
        else:
            self.register_parameter('bias',None) 

        self.reset_parameters()

    def reset_parameters(self): 
        glorot(self.lin.weight)
        
        glorot(self.a)
        zeros(self.bias)


    def forward(self, batch_mat, topk_edge, embedding): 

        x = self.lin(batch_mat) 
        x = (x, x)  

        out = self.propagate(topk_edge, x=x, embedding=embedding, topk_edge=topk_edge) 



        if self.bias is not None:
            out = out + self.bias  

        return out
        
        

    def message(self, x_i, x_j, edge_index_i, size_i, embedding, topk_edge):  
        
        embedding_j=torch.empty(size=x_i.shape)
        embedding_i=torch.empty(size=x_i.shape)   
        for i in range(len(topk_edge[0])): 
            j_idx, i_idx=topk_edge[0][i], topk_edge[1][i]
            embedding_j[i]=embedding[j_idx]
            embedding_i[i]=embedding[i_idx]

        git = torch.cat((x_i, embedding_i), dim=-1) 
        gjt = torch.cat((x_j, embedding_j), dim=-1) 


        gijt=torch.cat((git,gjt),dim=-1) 
        alpha= (self.a * gijt).sum(-1) 
        alpha = alpha.view(-1, 1)  


        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, size_i)


        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha


    def __repr__(self): 
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, 1)

