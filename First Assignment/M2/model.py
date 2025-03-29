import torch
import torch.nn as nn

from torch.distributions.normal import Normal

class Flow1d(nn.Module):
		def __init__(self, n_components):
				super(Flow1d, self).__init__()
				self.mus	= nn.Parameter(torch.randn(n_components), requires_grad=True)
				self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
				self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

		def forward(self, x):
				x = x.view(-1, 1)
				weights = self.weight_logits.softmax(dim=0).view(1, -1)
				distribution = Normal(self.mus, self.log_sigmas.exp())
				z = torch.sum(weights * distribution.cdf(x), dim=1, keepdim=True)
				dz_by_dx = torch.sum(weights * torch.exp(distribution.log_prob(x)), dim=1, keepdim=True)
				log_dz_by_dx = torch.log(dz_by_dx) # probably an epsilon is needed
				return z, log_dz_by_dx

class LogitTransform(nn.Module):
		def __init__(self, alpha):
				super(LogitTransform, self).__init__()
				self.alpha = alpha

		def forward(self, x):
				x_new = self.alpha / 2 +	(1 - self.alpha) * x
				z = torch.log(x_new) - torch.log(1 - x_new)
				dz_by_dx = (1 - self.alpha) * (1	/ x_new + 1 / (1 - x_new))
				log_dz_by_dx = torch.log(dz_by_dx) # probably an epsilon is needed
				return z, log_dz_by_dx

class FlowComposable1d(nn.Module):
		def __init__(self, flow_models_list):
				super(FlowComposable1d, self).__init__()
				self.flow_models_list = nn.ModuleList(flow_models_list)

		def forward(self, x):
				z = x
				total_dz_by_dx = 0
				for flow_model in self.flow_models_list:
						z, log_dz_by_dx = flow_model(z)
						total_dz_by_dx += log_dz_by_dx
				return z, total_dz_by_dx