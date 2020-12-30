import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from load_dataset import Dataset
import os
from train_model import MNIST_PAQ

if __name__ == '__main__':
	number_of_clients = 328
	aggregate_epochs = 10

	train_x, train_y, test_x, test_y = Dataset().load_csv()

	for local_epochs in [1, 3, 5, 7]:
		for r in [0.5, 0.6, 0.7, 0.8, 0.9]:
			for precision in [5, 6, 7]:
				filename = "saved_models_local_epochs_"+str(local_epochs)+"_r_"+str(r).replace('.', '_')+"_precision_"+str(precision)
				m = MNIST_PAQ(precision=precision, filename=filename, r=r, number_of_clients=number_of_clients, aggregate_epochs=aggregate_epochs, local_epochs=local_epochs)
				train_x, train_y = m.client_generator(train_x, train_y)
				m.train_aggregator({'x':train_x, 'y':train_y}, {'x':test_x, 'y':test_y})