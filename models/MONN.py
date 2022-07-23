import torch
from torch import nn
import torch.nn.functional as F
from models.base import *
import math
import numpy as np
from torch.nn.parameter import Parameter
atomf_len = 82
bondf_len = 6

def initialize_(input_dim, hidden_dim): # input_dim x hidden_dim의 weight
    weight = torch.nn.Parameter(torch.FloatTensor(input_dim, hidden_dim))
    torch.nn.init.kaiming_uniform_(weight)
    return weight

def embedding_(input, weight): # 원자 개수 x hidden_dim의 representation
    return F.leaky_relu(torch.mm(input, weight), 0.1)

class GCN(nn.Module):
    def __init__(self,batch_size=8, GNN_depth=2, k_head=3, hidden_dim1=10, hidden_dim2=10, data=0):
        super(GCN, self).__init__()

        self.GNN_depth = GNN_depth
        self.k_head = k_head
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.data = data
        self.batch_size = batch_size

        self.weight_init = initialize_(atomf_len, hidden_dim1)
        self.gather_weight = initialize_(self.hidden_dim1 + 6, self.hidden_dim1)
        self.update_weight = initialize_(self.hidden_dim1 * 2, self.hidden_dim1)

    def GraphConv_module(self):
        vertex_initial = torch.ones([10, 82])
        edge_initial = torch.ones([10, 6]) #Bond features are not updated.
        vertex_feature = embedding_(vertex_initial, self.weight_init) # 1 x h1
        super_node_init = torch.sum(vertex_initial, dim = 0, keepdim = True) # Summation of node features, 1 x h1
        super_node_feature = F.tanh(embedding_(super_node_init, self.super_node_weight)) # super_node_features, 1 x h1

        #t_i =

class Attention(nn.Module):
    def __init__(self, k_head, hidden_dim1) :
        super(Attention, self).__init__()
        self.k_head = k_head
        self.hidden_dim1 = hidden_dim1
    
        self.att_weight = initialize_(self.hidden_dim1,1)
        self.super_att_weight = initialize_(self.hidden_dim1, self.hidden_dim1)
        self.vertex_att_weight = initialize_(self.hidden_dim1, self.hidden_dim1)
        self.v_s_weight= initialize_(self.k_head * self.hidden_dim1, self.hidden_dim1)

    def forward(self, vertex, super_node):
        v_att = F.tanh(torch.mm(vertex,self.vertex_att_weight)) # 1 x hidden_dim1
        s_att = F.tanh(torch.mm(super_node,self.super_att_weight)) # 1 x hidden_dim1
        b_ = torch.mul(v_att, s_att) # 1 x hidden_dim1, elementwise multiplication
        alpha = F.softmax(torch.mm(b_,self.att_weight)) # 1 x 1, alpha의 차원

        concat = torch.mm(vertex, alpha) # 1 x k*h1
        output = torch.mm(concat, self.v_s_weight) # 1 x h1
        output = F.tanh(output)

        return output
class Warp_GRU(nn.Module):
    def __init__(self, k_head, hidden_dim1) : 
        super(Warp_GRU, self).__init__()
        self.hidden_dim1 = hidden_dim1
        self.k_head = k_head

        self.s_weight = initialize_(self.hidden_dim1, self.hidden_dim1)
        self.s_v_weight = initialize_(self.hidden_dim1, self.hidden_dim1)

        self.weight_11 = initialize_(self.hidden_dim1, self.hidden_dim1)
        self.weight_12 = initialize_(self.hidden_dim1, self.hidden_dim1)
        self.weight_21 = initialize_(self.hidden_dim1, self.hidden_dim1)
        self.weight_22 = initialize_(self.hidden_dim1, self.hidden_dim1)

        self.k_head_attn = Attention(self.k_head)

        self.v_gru = nn.GRU(self.hiddem_dim1, self.hidden_dim1)
        self.s_gru = nn.GRU(self.hidden_dim1, self.hidden_dim1)

    def forward(self, vertex, super_node, u_i):
        u_s = F.tanh(torch.mm(super_node, self.s_weight)) # 1 x h1
        u_s_v = F.tanh(torch.mm(super_node, self.s_v_weight)) # 1 x h1

        u_v_s = self.k_head_attn(vertex, super_node) # 1 x h1

        g_v_s = nn.Sigmoid(torch.mm(u_v_s, self.weight_11) + torch.mm(u_s, self.weight_12)) # 1 x h1
        t_v_s = torch.mul(1 - g_v_s, u_v_s) + torch.mul(g_v_s, u_s) # 1 x h1

        g_s_i = nn.Sigmoid(torch.mm(u_i, self.weight_21) + torch.mm(u_s_v, self.weight_22))
        t_s_i = torch.mul(1 - g_s_i,u_i) + torch.mul(g_s_i ,u_s_v)

        atom_node_updated = self.v_gru(vertex, t_s_i)
        super_node_updated = self.s_gru(super_node, t_v_s)

        return atom_node_updated, super_node_updated


class CNN(nn.Module):
    def __init__(self, num_layer, hidden_size, kernel_size):
        super(CNN, self).__init__()
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size

        self.init_word_features = initialize_(100,100)
        self.embed_seq = nn.Embedding(len(self.init_word_features), 20, padding_idx=0)
        #self.embed_seq.weight = nn.Parameter(self.init_word_features)
        self.embed_seq.weight.requires_grad = False

        self.conv_first = nn.Conv1d(20, self.hidden_size , kernel_size=self.kernel_size,
                                    padding=0)
        self.conv_last = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size,
                                   padding=0)

        self.plain_CNN = nn.ModuleList([])
        for i in range(self.num_layer):
            self.plain_CNN.append(nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size,
                                            padding=(self.kernel_size - 1) / 2))
    def forward(self, x):
        sequence = x
        embedding = self.embed_seq(sequence)
        x = F.leaky_relu(self.conv_first(embedding),0.1)
        for num in range(self.num_layer):
            x = self.plain_CNN[num](x)
            x = F.leaky_relu((x,0.1))
        x = F.leaky_relu(self.conv_last(x),0.1)

        return x

print(embedding_(torch.ones([2, 82]),initialize_(82,10)))
print(torch.nn.init.kaiming_uniform_(torch.FloatTensor(5, 5)))
a= GCN(8,3,12,10,20,0)
print(a.hidden_dim1)


att_weight = torch.ones([3*5,5])
print(att_weight)
b = torch.mul(torch.tensor([2,1,3]), torch.tensor([5,1,3]))
print(b)

print(CNN(5,10,5)(torch.LongTensor([1,2,3,4,5])))
