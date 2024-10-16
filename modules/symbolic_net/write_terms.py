import torch

def write_terms(individual):
    
    terms = []
    species = ['u', 'v']
 
    for term, param in zip(individual.poly_terms, individual.poly_params):
        string = f'{param:.3f}'
        for ind in term:
            string += f' * {species[ind]}'
        terms.append(string)
    
    for term, param, k, n in zip(individual.hill_terms, individual.hill_params_increasing, individual.hill_ks_increasing, individual.hill_ns_increasing):
        string = f'{param:.3f}'
        if len(term) == 1:
            string += f' * {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f})'
        else:     
            string += f' * {species[term[0]]} * {species[term[1]]}^{n:.3f} / (1 + {k:.3f} * {species[term[1]]}^{n:.3f})'
        terms.append(string)
    
    for term, param, k, n in zip(individual.hill_terms, individual.hill_params_decreasing, individual.hill_ks_decreasing, individual.hill_ns_decreasing):
        string = f'{param:.3f}'
        if len(term) == 1:
            string += f' * (1 - {species[term[0]]}^{n:.3f} / (1 + {k:.3f} * {species[term[0]]}^{n:.3f}))'
        else:     
            string += f' * {species[term[0]]} * (1 - {species[term[1]]}^{n:.3f} / (1 + {k:.3f} * {species[term[1]]}^{n:.3f}))'
        terms.append(string)
    
    return terms
