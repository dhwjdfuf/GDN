from traceback import print_tb
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import time
from torch_geometric.nn import GCNConv, GATConv, EdgeConv
import math
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential, BatchNorm1d, ReLU
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing 
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax

from torch_geometric.nn.inits import glorot, zeros
import time
from graph_layer import GraphLayer


class OutLayer(nn.Module): 
    def __init__(self, in_num, node_num, layer_num, inter_num = 512):
        super(OutLayer, self).__init__()

        modules = []

        for i in range(layer_num):
            if i == layer_num-1:
                modules.append(nn.Linear( in_num if layer_num == 1 else inter_num, 1))
            else:
                layer_in_num = in_num if i == 0 else inter_num
                modules.append(nn.Linear( layer_in_num, inter_num )) 
                modules.append(nn.BatchNorm1d(inter_num))
                modules.append(nn.ReLU())

        self.mlp = nn.ModuleList(modules) 

    def forward(self, x):
        out = x

        for mod in self.mlp:
            if isinstance(mod, nn.BatchNorm1d):
                out = out.permute(0,2,1) 
                out = mod(out)
                out = out.permute(0,2,1)
            else:
                out = mod(out)

        return out



class GNNLayer(nn.Module): 
    def __init__(self, in_channel, out_channel, inter_dim):
        super(GNNLayer, self).__init__()


        self.graph = GraphLayer(in_channel, out_channel, inter_dim=inter_dim)

        self.bn = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU()



        
    def forward(self, batch_mat,topk_edge, embedding): 

        out= self.graph(batch_mat, topk_edge, embedding)

        out = self.bn(out) 
        
        
        return self.relu(out) 



class GDN(nn.Module):
    def __init__(self, full_edges, node_num, embed_dim=64, out_layer_inter_dim=256, input_dim=10, out_layer_num=1, topk=20):
        super(GDN,self).__init__()


        self.embedding = nn.Embedding(node_num,embed_dim)

      
        self.bn_outlayer_in = nn.BatchNorm1d(embed_dim)  

        self.gnn_layer=GNNLayer(input_dim,embed_dim,2*embed_dim)
        
        self.topk = topk
        self.edges=full_edges

        self.out_layer = OutLayer(embed_dim, node_num, out_layer_num, inter_num = out_layer_inter_dim)
        self.dp = nn.Dropout(0.2)

    def init_params(self):
        nn.init.kaiming_uniform_(self.embedding.weight, a=math.sqrt(5))

    def forward(self, batch_tensor, org_edge_index): 

        batch_mat = batch_tensor.clone().detach() 
        
        batch_size, node_num, feature_len = batch_tensor.shape 

        batch_mat = batch_mat.view(-1, feature_len).contiguous() 
        scalar_data_num=batch_mat.shape[0] 
     
        edge_num = self.edges.shape[1] 

        all_embeddings = self.embedding(torch.arange(node_num)) 

        weights = all_embeddings.detach().clone() 
       
        cos_ji_mat = torch.matmul(weights, weights.T) 

        normed_mat = torch.matmul(weights.norm(dim=-1).view(-1,1), weights.norm(dim=-1).view(1,-1)) 
        cos_ji_mat = cos_ji_mat / normed_mat 

        topk_indices_ji = torch.topk(cos_ji_mat, self.topk, dim=-1)[1] 
        

        gated_i = torch.arange(0, node_num).T.unsqueeze(1).repeat(1, self.topk).flatten().unsqueeze(0) 
        gated_j = topk_indices_ji.flatten().unsqueeze(0) 
        topk_edge = torch.cat((gated_j, gated_i), dim=0) 

        gcn_out = self.gnn_layer(batch_mat, topk_edge, embedding=all_embeddings) 

        x = gcn_out.view(batch_size, node_num, -1) 

        indexes = torch.arange(0,node_num)
        out = torch.mul(x, self.embedding(indexes)) 
        
        out = out.permute(0,2,1)  
        out = F.relu(self.bn_outlayer_in(out))
        out = out.permute(0,2,1)

        out = self.dp(out)
        out = self.out_layer(out)
        out = out.view(-1, node_num)
   

        return out 


    


