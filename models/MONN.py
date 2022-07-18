import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from typing import Dict
from .base import *

    
class GraphConvolution(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.weight = Parameter(torch.FloatTensor(self.input_dim, self.hidden_dim))
        #self.bias = Parameter(torch.zeros(self.hidden_dim,))
        self.reset_parameters()
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        # output = torch.mm(input, self.weight) + self.bias
        return output
    
class GraphWarp(nn.Module):
    def __init__(self, hidden_dim, num_k):
        self.af = dict(tanh = nn.Tanh(),
                        sigmoid = nn.Sigmoid(),
                        softmax = nn.Softmax())
        self.hidden_dim = hidden_dim
        self.weight_s = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.weight_s_to_v = GraphConvolution(self.hidden_dim, self.hidden_dim)

        self.weight_gate11 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.weight_gate12 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.weight_gate21 = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.weight_gate22 = GraphConvolution(self.hidden_dim, self.hidden_dim)

        self.k_head_attn = Attention(num_k)

        self.v_gru = nn.GRU(self.hiddem_dim, self.hidden_dim)
        self.s_gru = nn.GRU(self.hidden_dim, self.hidden_dim)

    def forward(self, atom_node, super_node, updated_i):
        updated_s = self.weight_s(super_node)
        updated_s = self.af['tanh'](updated_s)
        updated_s_to_v = self.weight_s_to_v(super_node)
        updated_s_to_v = self.af['tanh'](updated_s_to_v)
        
        updated_v_to_s = self.k_head_attn(atom_node, super_node)
        
        g_v_to_s = nn.Sigmoid(self.weight_gate11(updated_v_to_s) + self.weight_gate12(updated_s))
        t_v_to_s = (1-g_v_to_s)*(updated_v_to_s) + g_v_to_s*(updated_s)

        g_s_to_i = nn.Sigmoid(self.weight_gate21(updated_i) + self.weight_gate22(updated_s_to_v))
        t_s_to_i = (1-g_s_to_i)*(updated_i) + g_s_to_i*(updated_s_to_v)

        atom_node_updated = self.v_gru(atom_node, t_s_to_i)
        super_node_updated = self.s_gru(super_node, t_v_to_s)
        
        return atom_node_updated, super_node_updated
    

#####torch.nn.MultiheadAttention
class Attention(nn.Module):
    def __init__(self, num_K):
        self.node_attn_weight = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.super_attn_weight = GraphConvolution(self.hidden_dim, self.hidden_dim)
        self.attn = GraphConvolution(self.hidden_dim, 1)
        self.weight_v_to_s = GraphConvolution(self.num_K*self.hidden_dim, self.hidden_dim) 

    def forward(self, node, super_node):
        node_attn = self.node_attn_weight(node)
        node_attn = nn.Tanh(node_attn)

        super_node_attn = self.super_attn_weight(super_node)
        super_node_attn = nn.Tanh(super_node_attn)

        b_ = torch.matmul(node_attn, super_node_attn)
        alpha = self.attn(b_)
        alpha = nn.Softmax(alpha)

        output = torch.mm(alpha, node)
        output = torch.sum(output, dim= 0)

        output = self.weight_v_to_s(output)
        output = nn.Tanh(output)

        return output
        
        
class GCN(nn.Module):
    def __init__(self, input_shape, hidden_dim, bond_feature, neighbor_info:Dict):
        super(GCN, self).__init__()
        #input_shape = (atom_features.shape, bond_features.shape)
        self.bond_feature = bond_feature
        self.nb_matrix = neighbor_info['neighbor_matrix']
        self.nb_atom = neighbor_info['atom']
        self.nb_bond = neighbor_info['bond']

        self.atom_dim = input_shape[0][-1]
        self.bond_dim = input_shape[1][-1]
        self.hidden_dim = hidden_dim
        
        self.gc_init = GraphConvolution(self.atom_dim, self.hidden_dim)
        self.gather_weight = GraphConvolution(self.hidden_dim+6, self.hidden_dim)
        self.update_weight = GraphConvolution(self.hidden_dim*2, self.hidden_dim)
        self.af = nn.LeakyReLU(negative_slope = 0.1)

        num_k = 3
        self.graph_warp_unit = GraphWarp(self.hidden_dim, num_k)

    def forward(self, input):
        atom_feature, atom_ids = input
        node_init, super_node_init = self.initialize(atom_feature)
        ############CODE L iterations(TO DO)###########
        #t_i(l), u_i(l) = message_passing_unit(v_i(l-1))
        t_i, u_i = self.message_passing_unit(node_init, atom_ids)
        #v_i(l), s_i(l) = graph_warp_unit(v_i(l-1), s_i(l-1), u_i =)
        updated_atom_node, updated_super_node  = self.graph_warp_unit(node_init, super_node_init, u_i)
        
        return updated_atom_node, updated_super_node

    def initialize(self, input):
        node_init = self.gc_init(input)
        node_init = self.af(node_init)
        super_node_init = torch.sum(node_init, dim = 0, keepdim = True)
        return node_init, super_node_init

    #torch.gather
    def gather_neighbor_info(self, atom_idx, atom_feature):
        neighbor_idx = torch.nonzero(self.nb_matrix[atom_idx])[-1].item()
        # print(neighbor_idx)
        atom_indices = self.nb_atom[atom_idx, :neighbor_idx+1]
        bond_indices = self.nb_bond[atom_idx, :neighbor_idx+1]

        nb_atoms_feat, nb_bonds_feat = [], []

        for indice in atom_indices:
            indice = indice.repeat(atom_feature.shape[1]).reshape(-1,1)
            nb_atoms_feat.append(torch.gather(atom_feature.T, 1, indice.cuda()).reshape(-1))
        nb_atoms_tensor = torch.cat(nb_atoms_feat).reshape(len(atom_indices), -1)

        for indice in bond_indices:
            nb_bonds_feat.append(self.bond_feature[indice])
        nb_bonds_tensor = torch.cat(nb_bonds_feat).reshape(len(bond_indices), -1)

        atom_local_info = torch.cat((nb_atoms_tensor, nb_bonds_tensor), dim = 1)
        # print(nb_atoms_tensor.shape, nb_bonds_tensor.shape, atom_local_info.shape)

        return atom_local_info

    
    def message_passing_unit(self, atom_feature, atom_ids):
        gathered_information = []
        for atom_id in range(torch.nonzero(atom_ids)[-1] +1 ):
            concatenated_neighbors = self.gather_neighbor_info(atom_id, )
            atom_output = self.gather_weight(concatenated_neighbors)
            atom_output = self.af(atom_output)
            atom_output = torch.sum(atom_output, dim =0, keepdim = True)
            gathered_information.append(atom_output)
        gathered_information = torch.cat(gathered_information)
        gathered_information = torch.cat((gathered_information, atom_feature[gathered_information.shape[0]:]), dim = 0)
        assert gathered_information.shape[0] == atom_feature.shape[0]
        #t_i = gathered_information
        #u_i = updated_information
        updated_information = torch.cat((gathered_information, atom_feature), dim = 1)
        updated_information = self.update_weight(updated_information)
        updated_information = self.af(updated_information)

        return gathered_information, updated_information
    



            
#Protein
# def make_linear_layers(self, input_dim, hidden_dim, output_dim, num_layers):
#     layers = []
#     activation_fn = nn.LeakyReLU(negative_slope = 0.1)
#     layers.append(nn.Linear(input_dim, hidden_dim),
#                     activation_fn)
#     for i in range(num_layers):
#         layers.append(nn.Linear(hidden_dim, hidden_dim))
#         layers.append(activation_fn)
#     return nn.Sequential(*layers)

class Block(nn.Module):
    def __init__(self, args, in_channels, out_channels, stride = 1, down_sample=False):
        super(Block, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size = args.kernel_size,
                                stride= stride, padding=1, bias = False)
        self.bn = nn.BatchNorm1d(out_channels)
        self.af = nn.LeakyReLU(negative_slope = 0.01)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(self, x):
        out = self.conv(x)
        # out = self.bn(out)
        out = self.af(x) #leakyrelu
        return out
    

class CNN(nn.Module):
    def __init__(self, args, block):
        super(CNN, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.layers = self.get_layers(block, args.in_channels, args.out_channels, stride =1)

        for m in self.modules():
            if isinstance(m, nn.conv1d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'leaky_relu')
            else:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def get_layers(self, block, in_channels, out_channels, stride=1):
        layer_list = nn.ModuleList([block(self.args, in_channels, out_channels , stride)])
        for _ in range(self.num_layers):
            layer_list.append(block(out_channels, out_channels)) 
        return nn.Sequential(*layer_list)

    def forward(self, x):
        out = self.layers(x)
        return out

def protein_cnn(args):
    block = Block
    model = CNN(args, block)
    return model



# class PairwiseInteraction(nn.Module):
#     def __init__(self, dims):
#         super(PairwiseInteraction, self).__init__()
#         atom = 0 #
#         residue = 1 #
        
#         atom_in, atom_out = dims[atom][0], dims[atom][1]
#         residue_in, residue_out = dims[residue][0], dims[residue][1]
        
#         self.atom_layer = nn.Linear(atom_in, atom_out)
#         self.residue_layer = nn.Linear(residue_in, residue_out)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, atom, residue):
#         atom = self.atom_layer(atom)
#         residue = self.residue_layer(residue)
#         inner_prod = torch.inner(atom, residue)
#         pred = self.sigmoid(inner_prod)
#         return pred

