import torch, pdb
import torch.nn as nn
import torchist

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.activations.softplus_relu import softplus_relu
from modules.utils.numpy_torch_conversion import to_torch
from modules.symbolic_net.individual import individual
from modules.genetic_algorithm.custom_deap_functions import (
    calculate_poly_terms, calculate_hill_terms)


class symbolic_net(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown reaction function. Includes 
    three hidden layers with 32 sigmoid-activated neurons. Output is linearly 
    activated to allow positive and negative reaction values.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u, v (torch tensor): predicted u and v values with shape (N, 2)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        F (torch tensor): predicted reaction values with shape (N, 1)
    '''
    
    def __init__(self, species, degree, coef_bounds, device, l1reg=0, nonzero_terms=0):
        
        super().__init__()
        self.F_min = -100
        self.F_max = 100
        self.F_weight = 1e10 / self.F_max
        
        self.param_min = -coef_bounds
        self.param_max = coef_bounds
        self.param_weight = 1e10 / self.param_max
        
        self.species = species
        self.degree = degree
        self.l1reg = l1reg
        self.nonzero_terms = nonzero_terms
        
        self.poly_terms = calculate_poly_terms(species, degree)
        self.hill_terms = calculate_hill_terms(species, degree)
        self.num_params = len(self.poly_terms) + 3 * 2 * len(self.hill_terms) 

        self.params = nn.Parameter(torch.rand(self.num_params).to(device))
        self.individual = individual(self.params, self.poly_terms, 
                                     self.hill_terms, self.num_params)
    
    def forward(self, input):
        output = self.individual.predict_f(input)
        return output
    
    def loss(self, pred, true):
        self.individual = individual(self.params, self.poly_terms, 
                                     self.hill_terms, self.num_params)
        self.MSE_loss = 0
        self.F_loss = 0
        self.param_loss = 0
        self.l1reg_loss = 0
        self.nonzero_terms_loss = 0
        
        # MSE loss
        self.MSE_loss += nn.functional.mse_loss(pred, true.view(-1))
        
        # Param loss
        self.F_loss += self.param_weight*torch.where(
            pred < self.F_min, (pred-self.F_min)**2, torch.zeros_like(pred))
        self.F_loss += self.param_weight*torch.where(
            pred > self.F_max, (pred-self.F_max)**2, torch.zeros_like(pred))

        self.param_loss += self.param_weight*torch.where(
            self.params < self.param_min, (self.params-self.param_min)**2, torch.zeros_like(self.params))
        self.param_loss += self.param_weight*torch.where(
            self.params > self.param_max, (self.params-self.param_min)**2, torch.zeros_like(self.params))
        
        # L1 Regularization
        if self.l1reg != 0:
            l1_norm = torch.norm(self.individual.poly_params, p=1) + torch.norm(self.individual.hill_params, p=1)
            self.l1reg_loss += self.l1reg * l1_norm
        
        # Nonzero term loss
        if self.nonzero_terms != 0:
            count = torch.count_nonzero(self.individual.poly_params) + torch.count_nonzero(self.individual.hill_params)
            self.nonzero_terms_loss += torch.abs(self.nonzero_terms - count) * self.nonzero_terms        

        total_loss = (self.MSE_loss + torch.mean(self.F_loss) + torch.mean(self.param_loss) + self.l1reg_loss + self.nonzero_terms_loss)
        return total_loss
    