import torch
import torch.nn as nn

@torch.no_grad()
def store_representations(self, input, output):
    self.representations.append(output.detach().cpu().numpy())
    return

@torch.no_grad()
def store_interactions(self, input, output):
    self.interactions.append(output.detach().cpu().numpy())
    return

@torch.no_grad()
def store_atomwise_compound_representations(self, input, output):
	self.atomwise_representations.append(output[-2].detach().cpu().numpy())
	self.compound_representations.append(output[-1].detach().cpu().numpy())
	return

@torch.no_grad()
def store_atomwise_resiwise_compound_representations(self, input, output):
	self.atomwise_representations.append(output[-3].detach().cpu().numpy())
	self.resiwise_representations.append(output[-2].detach().cpu().numpy())
	self.compound_representations.append(output[-1].detach().cpu().numpy())
	return


def mask_softmax(a, mask, dim=-1):
    a_max = torch.max(a, dim, keepdim=True)[0]
    a_exp = torch.exp(a - a_max)
    a_exp = a_exp * mask
    a_softmax = a_exp / (torch.sum(a_exp, dim, keepdim=True) + 1e-6)
    return a_softmax


class GraphDenseSequential(nn.Sequential):
    def __init__(self, *args):
        super(GraphDenseSequential, self).__init__(*args)

    def forward(self, X, adj, mask):
        for module in self._modules.values():
            try:
                X = module(X, adj, mask)
            except BaseException:
                X = module(X)

        return X