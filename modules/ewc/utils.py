from copy import deepcopy

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch.utils.data


def variable(t: torch.Tensor, use_cuda=True, **kwargs):
    if torch.cuda.is_available() and use_cuda:
        t = t.cuda()
    return Variable(t, **kwargs)


class EWC(object):
    def __init__(self, model: nn.Module, inputs: list, outputs: list, hist=None, edges=None, weight=1):

        self.model = model
        self.inputs = inputs
        self.outputs = outputs
        self.hist = hist
        self.edges = edges
        self.weight = int(weight)
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self._means = {}
        self._precision_matrices = self._diag_fisher()

        for n, p in deepcopy(self.params).items():
            self._means[n] = variable(p.data)

    def _diag_fisher(self):
        precision_matrices = {}
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            precision_matrices[n] = variable(p.data)

        self.model.eval()
        for input, output in zip(self.inputs, self.outputs):
            self.model.zero_grad()
            input = variable(input.unsqueeze(0))
            label = variable(output.unsqueeze(0))
            if self.hist is not None and self.edges is not None:
                loss = self.model.loss(input, label, self.hist, self.edges)
            else:
                loss = self.model.loss(input, label)
            loss.backward()

            for n, p in self.model.named_parameters():
                precision_matrices[n].data += p.grad.data ** 2 / len(self.inputs)
        
        for input, output in zip(self.inputs, self.outputs):
            self.model.zero_grad()
            input = variable(input.unsqueeze(0))
            label = variable(output.unsqueeze(0))

        precision_matrices = {n: p for n, p in precision_matrices.items()}
        return precision_matrices

    def penalty(self, model: nn.Module):
        loss = 0
        for n, p in model.named_parameters():
            _loss = self._precision_matrices[n] * (p - self._means[n]) ** 2
            loss += _loss.sum()
        return loss * self.weight