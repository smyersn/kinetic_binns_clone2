import torch, pdb
import torch.nn as nn
import torchist

from modules.binn.build_mlp import build_mlp
from modules.utils.gradient import gradient
from modules.activations.softplus_relu import softplus_relu
from modules.utils.numpy_torch_conversion import to_torch


class D_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the unknown diffusivity function. 
    Includes three hidden layers with 32 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted diffusivities non-negative.
    
    Inputs:
        input_features (int): number of input features
        scale        (float): input scaling factor
    
    Args:
        u (torch tensor): predicted u values with shape (N, 1)
        t (torch tensor): optional time values with shape (N, 1)
        
    Returns:
        D (torch tensor): predicted diffusivities with shape (N, 1)
    '''
    
    
    def __init__(self, input_features=2, layers=[32, 32, 32, 2]):
        
        super().__init__()
        self.inputs = input_features
        self.min = 0
        self.max = 100
        self.mlp = build_mlp(
            input_features=input_features, 
            layers=layers,
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=softplus_relu())
        
    def forward(self, uv, t=None):
        
        if t is None:
            D = self.mlp(uv)
        else:
            D = self.mlp(torch.cat([uv, t], dim=1))    
        D = self.max * D
        
        return D

class uv_MLP(nn.Module):
    
    '''
    Construct MLP surrogate model for the solution of the governing PDE. 
    Includes three hidden layers with 128 sigmoid-activated neurons. Output
    is softplus-activated to keep predicted species concentrations non-negative.
    
    Inputs:
        scale (float): output scaling factor, defaults to carrying capacity
    
    Args:
        inputs (torch tensor): x and t pairs with shape (N, 2)
        
    Returns:
        outputs (torch tensor): predicted u and v values with shape (N, 2)
    '''
    
    def __init__(self, layers=[128, 128, 128, 2]):
        
        super().__init__()
        self.mlp = build_mlp(
            input_features=2, 
            layers=layers,
            #layers=8*[128] + [2],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=softplus_relu())
    
    def forward(self, inputs):
        outputs = self.mlp(inputs)
        return outputs


class F_MLP(nn.Module):
    
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
    
    def __init__(self, layers=[32, 32, 32, 1]):
        
        super().__init__()
        self.inputs = 2
        self.min = -200 # max and min calculated with u = v = [0, 2]
        self.max = 200
        self.mlp = build_mlp(
            input_features=2, 
            layers=layers,
            #layers=8*[128]+[1],
            activation=nn.Sigmoid(), 
            linear_output=True)
    
    def forward(self, uv, t=None):
        
        if t is None:
            F = self.max * self.mlp(uv)
        else:
            F = self.max * self.mlp(torch.cat([uv, t], dim=1))
        #F = self.max * F
        return F
    
class BINN(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self, data=None, uv_layers=None, f_layers=None):
        
        super().__init__()
                
        # surface fitter
        if uv_layers is None:
            self.surface_fitter = uv_MLP()
        else:
            self.surface_fitter = uv_MLP(layers=uv_layers)
        
        # pde functions
        if f_layers is None:
            self.reaction = F_MLP()
        else:
            self.reaction = F_MLP(layers=f_layers)
        
        # parameter extrema
        self.F_min = self.reaction.min
        self.F_max = self.reaction.max
        
        # input extrema
        if data is not None:
            self.x_min = torch.min(data[:, 0])
            self.x_max = torch.max(data[:, 0])
            self.t_min = torch.min(data[:, 1]) 
            self.t_max = torch.max(data[:, 1])
        
        else:
            self.x_min = 0
            self.x_max = 10
            self.t_min = 0
            self.t_max = 25

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e0
        self.pde_weight = 1e0
        self.F_weight = 1e10 / self.F_max
        
        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 10000
        
        # model name
        self.name = 'Dumlp_Dvmlp_Fmlp'
    
    def forward(self, inputs):
        
        # cache input batch for pde loss
        self.inputs = inputs
        return self.surface_fitter(self.inputs)
    
    def gls_loss(self, pred, true, density_weight, hist, edges):
        
        residual = (pred - true)**2
                
        # proportional GLS weighting
        residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        if hist is not None and edges is not None:
            reciprocal_density = calc_reciprocal_density(true, hist, edges)           
            residual *= torch.tensor([[density_weight, density_weight]]).to(self.inputs.device) * torch.column_stack((reciprocal_density, reciprocal_density)).to(self.inputs.device)

        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, density_weight, hist, edges, return_mean=True):
        # unpack inputs
        x = inputs[:, 0][:,None]
        t = inputs[:, 1][:,None]

        # partial derivative computations 
        u = outputs[:, 0].clone()
        d1 = gradient(u, inputs, order=1)
        ux = d1[:, 0][:, None]
        ut = d1[:, 1][:, None]

        v = outputs[:, 1].clone()
        d2 = gradient(v, inputs, order=1)
        vx = d2[:, 0][:, None]
        vt = d2[:, 1][:, None]
        
        uv = torch.column_stack((u, v))
        u = u[:, None]
        v = v[:, None]

        # reaction
        if self.reaction.inputs == 2:
            F = self.reaction(uv)
        else:
            F = self.reaction(uv, t)

        # Reaction-diffusion equation
        LHS_u = ut
        RHS_u = gradient(0.01 * ux, inputs)[:, 0][:,None] + F
        LHS_v = vt
        RHS_v = gradient(1 * vx, inputs)[:, 0][:,None] - F
        pde_loss = (LHS_u - RHS_u)**2 + (LHS_v - RHS_v)**2
        
        if hist is not None and edges is not None:
            # add weight based on density
            reciprocal_density = calc_reciprocal_density(uv, hist, edges).to(self.inputs.device)
            pde_loss *= density_weight * reciprocal_density.unsqueeze(1)

        # constraints on learned parameters
        self.F_loss = 0
        self.F_loss += self.F_weight*torch.where(
            F < self.F_min, (F-self.F_min)**2, torch.zeros_like(F))
        self.F_loss += self.F_weight*torch.where(
            F > self.F_max, (F-self.F_max)**2, torch.zeros_like(F))
        
        # no derivative constraints included (Du and Dv constant, no obvious relationship for F)
        
        if return_mean:
            return torch.mean(pde_loss + self.F_loss)
        else:
            return pde_loss + self.F_loss
    
    def loss(self, pred, true, density_weight=0, hist=None, edges=None):
        self.gls_loss_val = 0
        self.pde_loss_val = 0
        
        # load cached inputs from forward pass
        inputs = self.inputs
        
        # compute surface loss
        self.gls_loss_val = self.surface_weight*self.gls_loss(pred, true, density_weight, hist, edges)
                
        # randomly sample from input domain for PDE loss
        x = torch.rand(self.num_samples, 1, requires_grad=True) 
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled locations
        outputs_rand = self.surface_fitter(inputs_rand)

        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand, density_weight, hist, edges)
        return self.gls_loss_val + self.pde_loss_val

class BINN_diff(nn.Module):
    
    '''
    Constructs a biologically-informed neural network (BINN) composed of
    cell density dependent diffusion and growth MLPs with an optional time 
    delay MLP.
    
    Inputs:
        delay (bool): whether to include time delay MLP
        
    
    '''
    
    def __init__(self):
        
        super().__init__()
        
        # surface fitter
        self.surface_fitter = uv_MLP()
        
        # diffusion
        self.diffusion = D_MLP()
        
        # pde functions
        self.reaction = F_MLP()
        
        # parameter extrema
        self.F_min = self.reaction.min
        self.F_max = self.reaction.max
        self.D_min = self.diffusion.min
        self.D_max = self.diffusion.max
        
        # input extrema
        self.x_min = 0 
        self.x_max = 10
        self.t_min = 0.0
        self.t_max = 24.5

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e0
        self.pde_weight = 1e0
        self.F_weight = 1e10 / self.F_max
        self.D_weight = 1e10 / self.D_max
        
        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 10000
        
        # model name
        self.name = 'Dumlp_Dvmlp_Fmlp'
    
    def forward(self, inputs):
        
        # cache input batch for pde loss
        self.inputs = inputs
        return self.surface_fitter(self.inputs)
    
    def gls_loss(self, pred, true):
        
        residual = (pred - true)**2
        
        # add weight to initial condition
        residual *= torch.where(self.inputs[:, 1][:, None]==0, 
                                self.IC_weight*torch.ones_like(pred), 
                                torch.ones_like(pred))
        
        # proportional GLS weighting
        residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, hist, edges, return_mean=True):
        # unpack inputs
        x = inputs[:, 0][:,None]
        t = inputs[:, 1][:,None]

        # partial derivative computations 
        u = outputs[:, 0].clone()
        d1 = gradient(u, inputs, order=1)
        ux = d1[:, 0][:, None]
        ut = d1[:, 1][:, None]

        v = outputs[:, 1].clone()
        d2 = gradient(v, inputs, order=1)
        vx = d2[:, 0][:, None]
        vt = d2[:, 1][:, None]
        
        uv = torch.column_stack((u, v))
        u = u[:, None]
        v = v[:, None]

        # reaction
        if self.reaction.inputs == 2:
            F = self.reaction(uv)
        else:
            F = self.reaction(uv, t)
            
        # diffusion
        Du, Dv = self.diffusion(uv).T

        # Reaction-diffusion equation
        LHS_u = ut
        RHS_u = gradient(Du * ux, inputs)[:, 0][:,None] + F
        LHS_v = vt
        RHS_v = gradient(Dv * vx, inputs)[:, 0][:,None] - F
        pde_loss = (LHS_u - RHS_u)**2 + (LHS_v - RHS_v)**2
        
        # constraints on learned parameters
        self.F_loss = 0
        self.F_loss += self.F_weight*torch.where(
            F < self.F_min, (F-self.F_min)**2, torch.zeros_like(F))
        self.F_loss += self.F_weight*torch.where(
            F > self.F_max, (F-self.F_max)**2, torch.zeros_like(F))
        
        self.D_loss = 0
        self.D_loss += self.D_weight*torch.where(
            Du < self.D_min, (Du-self.D_min)**2, torch.zeros_like(Du))
        self.D_loss += self.D_weight*torch.where(
            Du > self.D_max, (Du-self.D_max)**2, torch.zeros_like(Du))
        self.D_loss += self.D_weight*torch.where(
            Dv < self.D_min, (Dv-self.D_min)**2, torch.zeros_like(Dv))
        self.D_loss += self.D_weight*torch.where(
            Dv > self.D_max, (Dv-self.D_max)**2, torch.zeros_like(Dv))
        
        # no derivative constraints included (Du and Dv constant, no obvious relationship for F)
        
        if return_mean:
            return torch.mean(pde_loss + self.F_loss)
        else:
            return pde_loss + self.F_loss
    
    def loss(self, pred, true):
        self.gls_loss_val = 0
        self.pde_loss_val = 0
        
        # load cached inputs from forward pass
        inputs = self.inputs
        
        # compute surface loss
        self.gls_loss_val = self.surface_weight*self.gls_loss(pred, true)
                
        # randomly sample from input domain for PDE loss
        x = torch.rand(self.num_samples, 1, requires_grad=True) 
        x = x*(self.x_max - self.x_min) + self.x_min
        t = torch.rand(self.num_samples, 1, requires_grad=True)
        t = t*(self.t_max - self.t_min) + self.t_min
        inputs_rand = torch.cat([x, t], dim=1).float().to(inputs.device)

        # predict surface fitter at sampled locations
        outputs_rand = self.surface_fitter(inputs_rand)

        # compute PDE loss at sampled locations
        self.pde_loss_val += self.pde_weight*self.pde_loss(inputs_rand, outputs_rand)
        return self.gls_loss_val + self.pde_loss_val
    
    

def calc_reciprocal_density(uv, hist, edges):
    uv_indices = torch.zeros((len(uv), len(uv.T)))
    # iterate over columns
    for i in range(len(uv.T)):
        buckets = torch.bucketize(uv[:, i], edges[i], right=True)
        # convert from buckets to indices, correct so max value falls in last bucket
        indices = torch.where(buckets != 0, buckets-1, buckets)
        indices_corrected = torch.where(indices == len(edges[i])-1, indices-1, indices)
        uv_indices[:, i] = indices_corrected
        
    density = torch.tensor([hist[int(indices[0]), int(indices[1])] for indices in uv_indices])
    reciprocal_density = 1 / density
        
    return reciprocal_density