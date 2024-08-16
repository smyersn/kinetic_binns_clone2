import torch, pdb
import torch.nn as nn

from Modules.Models.BuildMLP import BuildMLP
from Modules.Utils.Gradient import Gradient
from Modules.Activations.SoftplusReLU import SoftplusReLU


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
    
    def __init__(self):
        
        super().__init__()
        self.mlp = BuildMLP(
            input_features=2, 
            layers=[128, 128, 128, 2],
            activation=nn.Sigmoid(), 
            linear_output=False,
            output_activation=SoftplusReLU())
    
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
    
    def __init__(self):
        
        super().__init__()
        self.inputs = 2
        self.min = -2 # max and min calculated with u = v = [0, 2]
        self.max = 6
        self.mlp = BuildMLP(
            input_features=2, 
            layers=[32, 32, 32, 1],
            activation=nn.Sigmoid(), 
            linear_output=True)
    
    def forward(self, uv, t=None):
        
        if t is None:
            F = self.mlp(uv)
        else:
            F = self.mlp(torch.cat([uv, t], dim=1))
        F = self.max * F
        return F
    
class BINN(nn.Module):
    
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
        
        # pde functions
        self.reaction = F_MLP()
        
        # parameter extrema
        self.F_min = self.reaction.min
        self.F_max = self.reaction.max
        self.K = 1.7e3
        
        # input extrema
        self.x_min = 0 
        self.x_max = 10
        self.t_min = 0.0
        self.t_max = 24.5

        # loss weights
        self.IC_weight = 1e1
        self.surface_weight = 1e0
        self.pde_weight = 1e0
        self.bc_weight = 1e0
        self.F_weight = 1e10 / self.F_max
        self.dFdu_weight = self.F_weight * self.K
        
        # proportionality constant
        self.gamma = 0.2

        # number of samples for pde loss
        self.num_samples = 10000
        
        # model name
        self.name = 'Dumlp_Dvmlp_Fmlp'
    
    def forward(self, inputs):
        
        # cache input and output batch for pde loss and bc loss
        self.inputs = inputs
        self.outputs = self.surface_fitter(self.inputs)
        return self.outputs
    
    def gls_loss(self, pred, true):
        
        residual = (pred - true)**2
        
        # add weight to initial condition
        residual *= torch.where(self.inputs[:, 1][:, None]==0, 
                                self.IC_weight*torch.ones_like(pred), 
                                torch.ones_like(pred))
        
        # proportional GLS weighting
        residual *= pred.abs().clamp(min=1.0)**(-self.gamma)
        
        return torch.mean(residual)
    
    def pde_loss(self, inputs, outputs, return_mean=True):
        # unpack inputs
        x = inputs[:, 0][:,None]
        t = inputs[:, 1][:,None]

        # partial derivative computations 
        u = outputs[:, 0].clone()
        d1 = Gradient(u, inputs, order=1)
        ux = d1[:, 0][:, None]
        ut = d1[:, 1][:, None]

        v = outputs[:, 1].clone()
        d2 = Gradient(v, inputs, order=1)
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
        RHS_u = Gradient(0.01 * ux, inputs)[:, 0][:,None] + F
        LHS_v = ut
        RHS_v = Gradient(1 * vx, inputs)[:, 0][:,None] - F
        pde_loss = (LHS_u - RHS_u)**2 + (LHS_v - RHS_v)**2

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
        
    def bc_loss(self, inputs, outputs):
        u = outputs[:, 0].clone()
        v = outputs[:, 1].clone()
        
        # define meshes from outputs of surface fitter
        v_mesh = torch.linspace(0, torch.max(v).item(), 501).to('cuda')
        
        # create tensors along axes of reaction domain
        zeros = torch.zeros(501).to('cuda')
        zero_u = torch.column_stack((zeros, v_mesh)).to('cuda')
        
        # calculate reaction values along axes calculate MSE vs. zero tensor
        zero_u_f = torch.flatten(self.reaction(zero_u))
        
        mse = nn.MSELoss()

        return 5 * mse(zero_u_f, zeros) 

    def loss(self, pred, true):
        self.gls_loss_val = 0
        self.pde_loss_val = 0
        self.bc_loss_val = 0
        
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

        # load cached outputs from forward pass
        outputs = self.outputs

        # compute BC loss along axes of reaction domain
        self.bc_loss_val += self.bc_weight*self.bc_loss(inputs, outputs)
        
        return self.gls_loss_val + self.pde_loss_val + self.bc_loss_val