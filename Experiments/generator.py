import numpy as np
import json
from keras.utils import Sequence
from PIL import Image
from skimage.io import imread


import os
import random

class Data_Gen(Sequence):
	def __init__(self, path, batch_size):
		self.filenames = [f for f in os.listdir(path)]
		random.shuffle(self.filenames)
		self.path = path
		self.batch_size = batch_size

	def __len__(self):
		return (len(self.filenames)) // self.batch_size

	def __getitem__(self, idx):
		batch_info = []
		batch_imgs = []
		batch_outcomes = []
		total_mal = 0
		for filename in [self.filenames[x] for x in range(idx*self.batch_size, (idx+1)*self.batch_size) if self.filenames[x][-4:] == ".jpg"]:
			try:
				batch_imgs.append(imread(self.path+filename))
				batch_info.append([int(filename.split(',')[1])//10, int(filename.split(',')[2])])

				total_mal += 1 if filename.split(',')[0] == '1' else 0
				batch_outcomes.append(int(filename.split(',')[0]))

			
			except FileNotFoundError:
					print("Missing file", filename)
					continue
			
			except KeyError:
					continue

		# print("\nMal%", total_mal/self.batch_size)
		return [np.array(batch_info), np.array(batch_imgs)], np.array(batch_outcomes)


class Data_Gen_CNN(Sequence):
	def __init__(self, path, batch_size):
		self.filenames = [f for f in os.listdir(path)]
		random.shuffle(self.filenames)
		self.path = path
		self.batch_size = batch_size

	def __len__(self):
		return (len(self.filenames)) // self.batch_size

	def __getitem__(self, idx):
		batch_imgs = []
		batch_outcomes = []
		total_mal = 0
		for filename in [self.filenames[x] for x in range(idx*self.batch_size, (idx+1)*self.batch_size) if self.filenames[x][-4:] == ".jpg"]:
			try:
				batch_imgs.append(imread(self.path+filename))

				total_mal += 1 if filename.split(',')[0] == '1' else 0
				batch_outcomes.append(int(filename.split(',')[0]))
			
			except FileNotFoundError:
					print("Missing file", filename)
					continue
			
			except KeyError:
					continue

		# print("\nMal%", total_mal/self.batch_size)
		return np.array(batch_imgs), np.array(batch_outcomes)


# def custom_generator(path, batch_size):
# 	i = 0
# 	filenames = os.listdir(path)
# 	while True:
# 		batch = {'images': [], 'features': [], 'labels': []}
# 		for b in range(batch_size):
# 			if i == len(filenames):
# 				i = 0
# 				random.shuffle(filenames)
			
# 			filename = filenames[i]
# 			batch['features'].append(np.array(float(filename.split(',')[1:3])))
# 			batch['labels'].append(np.array([1]) if filename.split(',')[0] == '1' else np.array([0]))
# 			batch['images'].append(resize(imread(path+'/'+filename), (224,224)))
# 			i += 1

# 		batch['images'] = np.array(batch['images'])
# 		batch['features'] = np.array(batch['features'])
# 		batch['labels'] = np.array([batch['labels']])
# 		yield [batch['images'], batch['features']], batch['labels']








