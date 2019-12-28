import os
import sys
import csv
import random

from skimage.io import imread, imsave
from skimage.transform import resize

random.seed(42)

# set locations and image size
mal_imgs_path = "../Data/images/malignant"
ben_imgs_path = "../Data/images/benign"
meta_data_csv = "../Data/metadata.csv"

img_height, img_width = 224, 224

# define split ratios
train_ratio = .8
val_ratio = .1
test_ratio = 1 - train_ratio - val_ratio

# get all image names in locations
mal_list = os.listdir(mal_imgs_path)
ben_list = os.listdir(ben_imgs_path)

# shuffle list of images
random.shuffle(mal_list)
random.shuffle(ben_list)

# get size of each
mal_len = len(mal_list)
ben_len = len(ben_list)

# define split indices
mal_train_split = int(train_ratio * mal_len)
ben_train_split = int(train_ratio * ben_len)

mal_val_split = int((train_ratio + val_ratio) * mal_len)
ben_val_split = int((train_ratio + val_ratio) * ben_len)

# split lists
mal_train = mal_list[:mal_train_split]
mal_val = mal_list[mal_train_split:mal_val_split]
mal_test = mal_list[mal_val_split:]

ben_train = ben_list[:ben_train_split]
ben_val = ben_list[ben_train_split:ben_val_split]
ben_test = ben_list[ben_val_split:]

train = mal_train + ben_train
val = mal_val + ben_val
test = mal_test + ben_test

# load csv
with open('../Data/metadata.csv', mode='r') as f:
    reader = csv.reader(f)
    metadata = {rows[0]:[rows[2], rows[3], rows[4]] for rows in reader if rows[2] != "" and rows[3] != "" and rows[4] != ""}

# print(metadata)

print("Data Sizes:")
print("Train benign:", len(ben_train))
print("Train malignant:", len(mal_train))
print("Validation benign:", len(ben_val))
print("Validation malignant:", len(mal_val))
print("Test benign:", len(ben_test))
print("Test malignant:", len(mal_test))

# set things to be done
# list_of_lists = zip(['benign', 'benign', 'benign', 'malignant', 'malignant', 'malignant'], ['train', 'val', 'test', 'train', 'val', 'test'], [ben_train, ben_val, ben_test, mal_train, mal_val, mal_test])

# list_of_lists = zip(['benign', 'benign', 'malignant', 'malignant'], ['val', 'test', 'val', 'test'], [ben_val, ben_test, mal_val, mal_test])
list_of_lists = zip(['benign', 'malignant'], ['train', 'train'], [ben_train, mal_train])



# copy images over
for img_type, section_type, section_split in list_of_lists:
	count = 0	
	for cur_img in section_split:
		source = "../Data/images/"+img_type+"/"+cur_img

		# check if metadata for image available, otherwise continue
		try:
			img_info = metadata[cur_img.split(".")[0]] 
		except:
			print("could not find data for:", cur_img.split(".")[0])
			continue

		# encode male:1 female:0
		if img_info[2] == 'male':
			target_name = img_info[0]+","+img_info[1]+",1,"+str(count)+".jpg"
		else:
			target_name = img_info[0]+","+img_info[1]+",0,"+str(count)+".jpg"

		target = "../Data/"+section_type+"/"+target_name

		print("img:", source, "| being saved as:", target_name)

		count += 1

		try:
			img = imread(source)
			imsave(target, resize(img, (224, 224)))
		except: 
			print("some form of error")







