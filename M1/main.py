from model import Flow1d
from torch.optim import Adam

import torch

# Calculates the negative log-likelihood loss
def loss_function(target_distribution, z, dz_by_dx):
		log_likelihood = target_distribution.log_prob(z) + torch.log(dz_by_dx)
		return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
		model.train()
		for batch in train_loader:
				z, dz_by_dx = model(batch)
				loss = loss_function(target_distribution, z, dz_by_dx)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

def eval_loss(model, data_loader, target_distribution):
		model.eval()
		total_loss = 0
		for batch in data_loader:
				z, dz_by_dx = model(batch)
				loss = loss_function(target_distribution, z, dz_by_dx)
				total_loss += loss * batch.size(0)
				return (total_loss / len(data_loader.dataset)).item()

def train_and_eval(epochs, lr, train_loader, test_loader, target_distribution, n_components=5):
		flow = Flow1d(n_components=n_components)
		optimizer = Adam(flow.parameters(), lr=lr)
		train_losses, test_losses = [], []
		for epoch in range(epochs):
				train(flow, train_loader, optimizer, target_distribution)
				train_losses.append(eval_loss(flow, train_loader, target_distribution))
				test_losses.append(eval_loss(flow, test_loader, target_distribution))
				return flow, train_losses, test_losses
