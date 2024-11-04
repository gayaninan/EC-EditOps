import pandas as pd
import numpy as np
from scipy.stats import dirichlet, multinomial, poisson

# Estimates the posterior probabilities of substitution, deletion, and insertion operations using a Dirichlet distribution.
def estimate_posterior_probabilities(file_path, alpha_prior=None):

    if alpha_prior is None:
        alpha_prior = [1, 1, 1]  # Default uniform prior

    data = pd.read_csv(file_path)

    observed_counts = [data['sub'].sum(), data['del'].sum(), data['ins'].sum()]

    posterior_alpha = [obs_count + prior for obs_count, prior in zip(observed_counts, alpha_prior)]

    posterior_dist = dirichlet(posterior_alpha)

    posterior_means = posterior_dist.mean()  #mean of the posterior distribution (the expected probability)

    return tuple(posterior_means)

def calculate_multinomial_probabilities(file_path):
    data = pd.read_csv(file_path)

    count_sub = data['sub'].sum()
    count_del = data['del'].sum()
    count_ins = data['ins'].sum()
    total_operations = count_sub + count_del + count_ins

    counts_ins = [count_ins, total_operations - count_ins]
    counts_del = [count_del, total_operations - count_del]
    counts_sub = [count_sub, total_operations - count_sub]

    p_ins = multinomial.pmf(counts_ins, total_operations, [count_ins/total_operations, 1 - count_ins/total_operations])
    p_del = multinomial.pmf(counts_del, total_operations, [count_del/total_operations, 1 - count_del/total_operations])
    p_sub = multinomial.pmf(counts_sub, total_operations, [count_sub/total_operations, 1 - count_sub/total_operations])

    probabilities = [p_sub, p_del, p_ins]

    return tuple(probabilities)

def calculate_relative_frequencies(file_path):

    data = pd.read_csv(file_path)

    count_sub = data['sub'].sum()
    count_del = data['del'].sum()
    count_ins = data['ins'].sum()
    total_operations = count_sub + count_del + count_ins

    p_sub = count_sub / total_operations
    p_del = count_del / total_operations
    p_ins = count_ins / total_operations

    probabilities = [p_sub, p_del, p_ins]

    return tuple(probabilities)


def calculate_poisson_probabilities(file_path):

    data = pd.read_csv(file_path)

    count_sub = data['sub'].sum()
    count_del = data['del'].sum()
    count_ins = data['ins'].sum()
    num_records = len(data)

    mean_sub = count_sub / num_records
    mean_del = count_del / num_records
    mean_ins = count_ins / num_records

    p_sub = poisson.pmf(count_sub, mean_sub)
    p_del = poisson.pmf(count_del, mean_del)
    p_ins = poisson.pmf(count_ins, mean_ins)

file_path_1 = 'analysis/outs/babylon_all.csv'  
file_path_2 = 'analysis/outs/kaggle_all.csv'
file_path_3 = 'analysis/outs/gcd_all.csv'  
file_path_4 = 'combined_all.csv'  

# def print_probs(path_1, path_2, path_3, path_4 , dist_method):
#     p_sub1, p_del1, p_ins1 = dist_method(path_1)
#     print(f"Babylon - p_sub: {p_sub1}, p_del: {p_del1}, p_ins: {p_ins1}")

#     p_sub2, p_del2, p_ins2 = dist_method(path_2)
#     print(f"Kaggle - p_sub: {p_sub2}, p_del: {p_del2}, p_ins: {p_ins2}")

#     p_sub3, p_del3, p_ins3 = dist_method(path_3)
#     print(f"GCD - p_sub: {p_sub3}, p_del: {p_del3}, p_ins: {p_ins3}")

#     p_sub4, p_del4, p_ins4 = dist_method(path_4)
#     print(f"mixed - p_sub: {p_sub4}, p_del: {p_del4}, p_ins: {p_ins4}")

# print_probs(file_path_1, file_path_2, file_path_3, file_path_4, estimate_posterior_probabilities)
# print_probs(file_path_1, file_path_2, file_path_3, calculate_multinomial_probabilities)
# print_probs(file_path_1, file_path_2, file_path_3, calculate_relative_frequencies)
print("")

p_sub1, p_del1, p_ins1 = estimate_posterior_probabilities(file_path_4)
print("sub: " + str(p_sub1)+" del: " + str( p_del1)+" ins: " + str( p_ins1))