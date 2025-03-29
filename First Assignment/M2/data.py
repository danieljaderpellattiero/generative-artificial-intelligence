import numpy as np
import torch.utils.data as data

def generate_mixture_of_gaussians(num_of_points):
		n = num_of_points // 3
		gaussian_1 = np.random.normal(-1, 0.25, (n, ))
		gaussian_2 = np.random.normal(1.5, 0.35, (n, ))
		gaussian_3 = np.random.normal(0.0, 0.2, (num_of_points-2*n, ))
		return np.concatenate([gaussian_1, gaussian_2, gaussian_3])

class NumpyDataset(data.Dataset):
	def __init__(self, array):
			super().__init__()
			self.array = array

	def __len__(self):
			return len(self.array)

	def __getitem__(self, index):
			return self.array[index]

n_train, n_test = 2000, 1000
train_data = generate_mixture_of_gaussians(n_train)
test_data = generate_mixture_of_gaussians(n_test)

train_loader = data.DataLoader(NumpyDataset(train_data), batch_size=128, shuffle=True)
test_loader = data.DataLoader(NumpyDataset(test_data), batch_size=128, shuffle=True)
