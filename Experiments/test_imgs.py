from skimage.io import imread
import os

path = "../Data/test/"
filenames = os.listdir(path)

for filename in filenames:
	if filename[-4:] == ".jpg":
		try:
			img = imread(path+filename)
		except:
			print("did not work for:", path+filename)