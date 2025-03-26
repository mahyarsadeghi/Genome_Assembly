'''
  Empirical KL divergence calculator
  By Yue Zhang, Nov 15, 2024
'''
import torch
from torch import Tensor
from torch import nn
from torch import optim

from gfn.utils.evaluation import compute_KL, calc_KL_using_model, PhiFunction


# Test the performance
if __name__ == "__main__":
    batch_size = 10068
    n_features = 17
    num_epochs = 200

    # KL-divergence from two samples of the same normal distribution
    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.randn(batch_size, n_features)

    # Compute empirical KL
    kl, phi = compute_KL(sampleA, sampleB, num_epochs=num_epochs)
    print(f"Empirical KL from two samples of the same normal distribution = {kl.item():.4f}")

    # Sample from the same distribution and calculate KL
    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.randn(batch_size, n_features)
    kl = calc_KL_using_model(phi, sampleA, sampleB, no_grad=True)
    print(f"Empirical KL from another two samples of the same distribution = {kl.item():.4f}")

    # KL-divergence from two different distributions
    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.rand(batch_size, n_features)
    kl, phi = compute_KL(sampleA, sampleB, num_epochs=num_epochs)
    print(f"Empirical KL from two samples of different distributions = {kl.item():.4f}")

    sampleA = torch.randn(batch_size, n_features)
    sampleB = torch.rand(batch_size, n_features)
    kl = calc_KL_using_model(phi, sampleA, sampleB, no_grad=True)
    print(f"Empirical KL from another two samples = {kl.item():.4f}")
