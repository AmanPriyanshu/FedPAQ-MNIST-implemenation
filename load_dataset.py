import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

class Dataset:
	def __init__(self, iid=True, batch_size=128):
		self.train_x = None
		self.train_y = None
		self.test_x = None
		self.batch_size = batch_size
		self.iid = iid

	def load_csv(self, path='./digits_dataset/'):
		batch_size = self.batch_size
		data = pd.read_csv(path + 'train.csv')
		data = data.values
		np.random.shuffle(data)
		self.train_x = np.array([np.reshape(i, (1, int(np.sqrt(i.shape[0])), int(np.sqrt(i.shape[0])))) for i in data.T[1:].T], dtype=np.float32)
		self.train_y = data.T[0]
		if self.iid == False:
			indexes = np.argsort(self.train_y)
			self.train_x, self.train_y = self.train_x[indexes], self.train_y[indexes]
		data = pd.read_csv(path + 'test.csv')
		data = data.values
		self.test_x = np.array([np.reshape(i, (1, int(np.sqrt(i.shape[0])), int(np.sqrt(i.shape[0])))) for i in data], dtype=np.float32)
		data = pd.read_csv(path + 'test_results.csv')
		data = data.values
		self.test_y = np.array(data.T[0])

		self.train_x = np.array([self.train_x[n:n+batch_size] for n in range(0, len(self.train_x)-batch_size, batch_size)])/255.0
		self.test_x = np.array([self.test_x[n:n+batch_size] for n in range(0, len(self.test_x)-batch_size, batch_size)])/255.0
		self.train_y = np.array([self.train_y[n:n+batch_size] for n in range(0, len(self.train_y)-batch_size, batch_size)])
		self.test_y = np.array([self.test_y[n:n+batch_size] for n in range(0, len(self.test_y)-batch_size, batch_size)])

		self.train_x, self.train_y, self.test_x, self.test_y = torch.from_numpy(self.train_x), torch.from_numpy(self.train_y), torch.from_numpy(self.test_x), torch.from_numpy(self.test_y)

		return self.train_x, self.train_y, self.test_x, self.test_y