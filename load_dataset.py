import pandas as pd
import numpy as np
import torch

class Dataset:
	def __init__(self):
		self.train_x = None
		self.train_y = None
		self.test_x = None

	def load_csv(self, path='./digits_dataset/'):
		data = pd.read_csv(path + 'train.csv')
		data = data.values
		self.train_x = np.array([np.reshape(i, (1, int(np.sqrt(i.shape[0])), int(np.sqrt(i.shape[0])))) for i in data.T[1:].T], dtype=np.float32)
		self.train_y = data.T[0]
		data = pd.read_csv(path + 'test.csv')
		data = data.values
		self.test_x = np.array([np.reshape(i, (1, int(np.sqrt(i.shape[0])), int(np.sqrt(i.shape[0])))) for i in data], dtype=np.float32)
		self.train_x, self.train_y, self.test_x = torch.from_numpy(self.train_x), torch.from_numpy(self.train_y), torch.from_numpy(self.test_x)
		return self.train_x, self.train_y, self.test_x

class MNIST_PAQ:
	def __init__(self):
		self.model = None

	def define_model(self):
		self.model = torch.nn.Sequential(
			torch.nn.Conv2d(1, 2, kernel_size=5),
			torch.nn.ReLU(),
			torch.nn.Conv2d(2, 4, kernel_size=7),
			torch.nn.ReLU(),
			torch.nn.Flatten(),
			torch.nn.Linear(1296, 512),
			torch.nn.ReLU(),
			torch.nn.Linear(512, 128),
			torch.nn.ReLU(),
			torch.nn.Linear(128, 32),
			torch.nn.ReLU(),
			torch.nn.Linear(32, 10),
			torch.nn.Sigmoid(),
		)		

	def get_weights(self, dtype=np.float32, precision=7):
		assert precision<8, "Floats generally have 6-7 significant digits. Please try a lower number than or equal to 7"
		weights = []
		for layer in self.model:
			try:
				weights.append([np.around(layer.weight.detach().numpy().astype(dtype), decimals=precision), np.around(layer.bias.detach().numpy().astype(dtype), decimals=precision)])
			except:
				continue
		return np.array(weights)

	def set_weights(self, weights):
		index = 0
		for layer_no, layer in enumerate(self.model):
			try:
				_ = self.model[layer_no].weight
				self.model[layer_no].weight = torch.nn.Parameter(weights[index][0])
				self.model[layer_no].bias = torch.nn.Parameter(weights[index][1])
				index += 1
			except:
				continue

	def average_weights(self, all_weights):
		all_weights = np.array(all_weights)
		all_weights = np.mean(all_weights, axis=0)
		all_weights = [[torch.from_numpy(i[0].astype(np.float32)), torch.from_numpy(i[1].astype(np.float32))] for i in all_weights]
		return all_weights

	def single_client(self, dataset, weights, E):
		self.define_model()
		if weights is not None:
			self.set_weights(weights)
		for epoch in range(E):
			for batch_x, batch_y in zip(dataset['x'], dataset['y']):
				y_pred = self.model(batch_x)


		pass

	def single_aggregator_epoch(self, datasets, clients):
		for dataset, client in zip(datasets, clients):
			print(self.model(dataset['x'])[0])
			weights_1 = self.get_weights()
			weights_2 = self.get_weights()
			all_weights = [weights_1, weights_2]
			weights = self.average_weights(all_weights)
			self.set_weights(weights)
			print(self.model(dataset['x'])[0])

		
		

train_x, train_y, test_x = Dataset().load_csv()

m = MNIST_PAQ()
m.single_client({'x':test_x, 'y':train_y}, None, 5)