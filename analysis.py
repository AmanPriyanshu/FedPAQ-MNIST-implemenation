import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from load_dataset import Dataset
import os
import time

class Test:
	def __init__(self, train_x, train_y, test_x, test_y):
		self.train_x, self.train_y, self.test_x, self.test_y = train_x, train_y, test_x, test_y
		self.criterion = torch.nn.CrossEntropyLoss()
		
	def single_model(self, path):
		model = torch.load(path)
		model.eval()

		with torch.no_grad():

			train_loss, train_acc = 0, 0

			for batch_x, target in zip(self.train_x, self.train_y):
				output = model(batch_x)
				loss = self.criterion(output, target)
				output = torch.argmax(output, dim=1)
				acc = torch.sum(output == target)/target.shape[0]
				loss = loss.item()
				acc = acc.item()
				train_loss += loss
				train_acc += acc

			train_loss, train_acc = train_loss/self.train_x.shape[0], train_acc/self.train_x.shape[0]

			test_loss, test_acc = 0, 0

			for batch_x, target in zip(self.test_x, self.test_y):
				output = model(batch_x)
				loss = self.criterion(output, target)
				output = torch.argmax(output, dim=1)
				acc = torch.sum(output == target)/target.shape[0]
				loss = loss.item()
				acc = acc.item()
				test_loss += loss
				test_acc += acc

		test_loss, test_acc = test_loss/self.test_x.shape[0], test_acc/self.test_x.shape[0]

		return [train_loss, train_acc, test_loss, test_acc]

	def analyse_type(self, filename):
		model_names = os.listdir(filename)
		models = np.array([filename+i for i in model_names])
		model_names_index = np.argsort(np.array([int(i[len('model_epoch_'):-3]) for i in model_names]))
		models = models[model_names_index]
		performance = []
		for model in models:
			performance.append(self.single_model(model))
		performance = pd.DataFrame(np.array(performance))
		performance.columns = ['train_loss', 'train_acc', 'test_loss', 'test_acc']
		performance.to_csv('./results/performance_metrics/'+filename[len('./results/models/'):-1]+'.csv', index=False)

		performance = performance.values
		best_index = np.argmin(performance.T[2])
		return performance[best_index]

	def analyse_all(self):
		bar = tqdm(total=3*4*3)
		all_best_performances = []
		with bar:
			for local_epochs in [1, 3, 5]:
				for r in [0.5, 0.667, 0.833, 1.0]:
					for precision in [5, 6, 7]:
						filename = "./results/models/saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(precision)+"/"
						best = self.analyse_type(filename)
						best = [local_epochs, r, precision] + [i for i in best]
						all_best_performances.append(best)
						bar.update(1)
		all_best_performances = pd.DataFrame(np.array(all_best_performances))
		all_best_performances.columns = ['local_epochs', 'r', 'precision', 'train_loss', 'train_acc', 'test_loss', 'test_acc']
		all_best_performances.to_csv('./results/best_performances.csv', index=False)


if __name__ == '__main__':
	train_x, train_y, test_x, test_y = Dataset().load_csv()
	
	test = Test(train_x, train_y, test_x, test_y)
	test.analyse_all()