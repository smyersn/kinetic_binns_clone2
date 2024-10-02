import sys, os
file_dir = os.path.dirname(os.path.realpath(__file__))
repo_start = f'{file_dir}/../../'
sys.path.append(repo_start)

from modules.utils.imports import *
from modules.genetic_algorithm.custom_deap_functions import (
    calculate_poly_terms, calculate_hill_terms)

class individual():   
    def __init__(self, params, poly_terms, hill_terms):
        self.params = params
        self.poly_terms = poly_terms
        self.hill_terms = hill_terms

        # Define parameter indices
        hill_params_start = len(self.poly_terms)
        hill_ks_start = hill_params_start + 2 * len(self.hill_terms)
        hill_ns_start = hill_ks_start + 2 * len(self.hill_terms)
        
        # Unpack individual's parameters
        self.poly_params = self.params[:hill_params_start]

        self.hill_params = self.params[hill_params_start:hill_ks_start]
        self.hill_params_increasing = self.hill_params[:len(self.hill_params) // 2]
        self.hill_params_decreasing = self.hill_params[len(self.hill_params) // 2:]

        self.hill_ks = self.params[hill_ks_start:hill_ns_start]
        self.hill_ks_increasing = self.hill_ks[:len(self.hill_ks) // 2]
        self.hill_ks_decreasing = self.hill_ks[len(self.hill_ks) // 2:]
    
        self.hill_ns = self.params[hill_ns_start:]
        self.hill_ns_increasing = self.hill_ns[:len(self.hill_ns) // 2]
        self.hill_ns_decreasing = self.hill_ns[len(self.hill_ns) // 2:]

    def predict_f(self, uv):
        # Contribution to pred from polynomial terms
        poly_pred = torch.zeros(len(uv), len(self.poly_terms)).to(uv.device)
        for i, term, param in zip(range(len(self.poly_terms)), self.poly_terms, self.poly_params):
            poly_pred[:, i] = param * torch.prod(uv[:, term], dim=1)
        
        # Contriubtion to pred from increasing Hill function terms
        inc_hill_pred = torch.zeros(len(uv), len(self.hill_terms)).to(uv.device)
        for i, term, param, k, n in zip(range(len(self.hill_terms)), self.hill_terms, self.hill_params_increasing, self.hill_ks_increasing, self.hill_ns_increasing):
            if len(term) == 1:
                inc_hill_pred[:, i] = param * (uv[:, term[0]]**n / (1 + k * uv[:, term[0]]**n))
            else:
                inc_hill_pred[:, i] = param * ((uv[:, term[0]] * uv[:, term[1]]**n) / (1 + k * uv[:, term[1]]**n))

        # Contriubtion to pred from decreasing Hill function terms
        dec_hill_pred = torch.zeros(len(uv), len(self.hill_terms)).to(uv.device)
        for i, term, param, k, n in zip(range(len(self.hill_terms)), self.hill_terms, self.hill_params_decreasing, self.hill_ks_decreasing, self.hill_ns_decreasing): 
            if len(term) == 1:
                dec_hill_pred[:, i] = param * (1 - (uv[:, term[0]]**n / (1 + k * uv[:, term[0]]**n)))
            else:
                dec_hill_pred[:, i] = param * ((uv[:, term[0]] * (1 - (uv[:, term[1]]**n) / (1 + k * uv[:, term[1]]**n))))
       
        return torch.sum(poly_pred, dim=1) + torch.sum(inc_hill_pred, dim=1) + torch.sum(dec_hill_pred, dim=1)
    
    def calculate_terms(self, uv):       
        # Create vector to store surface prediction for learned function
        total_surface = np.zeros(len(uv))

        # Contribution to surface from polynomial terms
        for term, param in zip(self.poly_terms, self.poly_params):
            term_surface = np.repeat(param, len(uv))
            for idx in term:
                term_surface *= uv[:, idx]
            total_surface += term_surface
        
        # Contriubtion to surface from increasing Hill function terms
        for term, param, k, n in zip(self.hill_terms, self.hill_params_increasing, self.hill_ks_increasing, self.hill_ns_increasing):
            term_surface = np.repeat(param, len(uv)) 
            if len(term) == 1:
                term_surface *= (uv[:, term[0]]**n / (1 + k * uv[:, term[0]]**n))
            else:
                term_surface *= ((uv[:, term[0]] * uv[:, term[1]]**n) / (1 + k * uv[:, term[1]]**n))
            total_surface += term_surface
    
        # Contriubtion to surface from decreasing Hill function terms
        for term, param, k, n in zip(self.hill_terms, self.hill_params_decreasing, self.hill_ks_decreasing, self.hill_ns_decreasing): 
            term_surface = np.repeat(param, len(uv))
            if len(term) == 1:
                term_surface *= 1 - (uv[:, term[0]]**n / (1 + k * uv[:, term[0]]**n))
            else:
                term_surface *= (uv[:, term[0]] * (1 - (uv[:, term[1]]**n) / (1 + k * uv[:, term[1]]**n)))
            total_surface += term_surface
        
        # Reshape to match mesh dimensions    
        total_surface = np.reshape(total_surface, (101, 101))

        return total_surface

    def calculate_surface(self) = 