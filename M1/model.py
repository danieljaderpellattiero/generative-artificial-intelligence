import torch
import torch.nn as nn

from torch.nn.functional import one_hot
from torch.distributions import Categorical
from torch.distributions.normal import Normal

class Flow1d(nn.Module):
				def __init__(self, n_components):
								super(Flow1d, self).__init__()
								self.n_components = n_components
								self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
								self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
								self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

	# Forward transformation - from X (complex, Gaussian mixture) to Z (standard normal distribution)
				def forward(self, x):
								x = x.view(-1,1)
								weights = self.weight_logits.softmax(dim=0).view(1,-1)
								distribution = Normal(self.mus, self.log_sigmas.exp())
								z = (weights * distribution.cdf(x)).sum(dim=1, keepdim=True)
								dz_by_dx = (weights * distribution.log_prob(x).exp()).sum(dim=1, keepdim=True)
								return z, dz_by_dx

				# Inverse transformation - from Z (standard normal distribution) to X (complex, Gaussian mixture)
				def generate(self, z):
								z = z.view(-1, 1)
								weights = self.weight_logits.softmax(dim=0)
								distribution_categorical = Categorical(weights)
								with torch.no_grad():
												comp_indexes = distribution_categorical.sample((z.shape[0],))
												comp_mask = one_hot(comp_indexes, num_classes=self.n_components).to(weights.dtype)
												sel_mus = torch.sum(comp_mask * self.mus.view(1, -1), dim=1, keepdim=True)
												sel_sigmas = torch.sum(comp_mask * self.log_sigmas.exp().view(1, -1), dim=1, keepdim=True)
												x_hat = Normal(sel_mus, sel_sigmas).icdf(z)
								return x_hat.view(-1, 1)
