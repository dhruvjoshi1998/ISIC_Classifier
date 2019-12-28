import keras
from keras.applications import VGG16, ResNet50V2, InceptionResNetV2
from keras.preprocessing import image
from keras.models import Model, Sequential, load_model
from keras.layers import Flatten, Dense, Dropout, GlobalAveragePooling2D, concatenate, Conv2D
from keras import backend as K
from keras.optimizers import Adam
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

import generator

img_height, img_width = 224, 224
train_data_dir = "Data/train"
validation_data_dir = "Data/val"

batch_sz = 32
num_epochs = 20 


def load_vgg16():
	pretrained_model = VGG16(weights="imagenet", include_top=False, input_shape = (img_width, img_height, 3))
	return pretrained_model

def load_inception_net():
	pretrained_model = InceptionResNetV2(weights="imagenet", include_top=False, input_shape = (img_width, img_height, 3))
	return pretrained_model

def load_res_net():
	pretrained_model = ResNet50V2(weights="imagenet", include_top=False, input_shape = (img_width, img_height, 3))
	return pretrained_model

def make_mlp():
	model = Sequential()
	model.add(Dense(10, activation = 'relu', input_shape = (2,)))
	model.add(Dropout(0.25))
	model.add(Dense(10, activation = 'relu'))
	return model

def make_full_model():
	pretrained_cnn = load_vgg16()
	mlp = make_mlp()

	for layer in pretrained_cnn.layers[:]:
	    layer.trainable = False

	x = pretrained_cnn.output
	x = Flatten()(x)
	x = Dense(256, activation="relu")(x)

	x = concatenate([mlp.output, x])

	x = Dense(256, activation="relu")(x)
	predictions = Dense(1, activation="sigmoid")(x)

	# creating the final model 
	model = Model(input=[mlp.input, pretrained_cnn.input], output=predictions)

	return model

def make_cnn_model():
	pretrained_cnn = load_vgg16()

	for layer in pretrained_cnn.layers[:]:
	    layer.trainable = False

	x = pretrained_cnn.output
	x = Flatten()(x)
	x = Dense(256, activation="relu")(x)
	x = Dense(256, activation="relu")(x)
	predictions = Dense(1, activation="sigmoid")(x)

	model = Model(input=pretrained_cnn.input, output=predictions)

	return model


def train_mixed_model():
	model = make_full_model()
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
	model.summary()

	mc = keras.callbacks.ModelCheckpoint('../Models/mixed_model_at_{epoch:03d}.h5', period=2)

	model.fit_generator(generator.Data_Gen("../Data/train/", batch_sz), epochs=num_epochs, verbose=1, validation_data=generator.Data_Gen("../Data/val/", 64), callbacks=[mc])

	model.save("../Models/mixed_model.h5")

def train_cnn_model():
	model = make_cnn_model()
	model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, beta_1=0.9, beta_2=0.999), metrics=['accuracy'])
	model.summary()

	mc = keras.callbacks.ModelCheckpoint('../Models/cnn_only_model_at_{epoch:03d}.h5', period=2)

	model.fit_generator(generator.Data_Gen_CNN("../Data/train/", batch_sz), epochs=num_epochs, verbose=1, validation_data=generator.Data_Gen_CNN("../Data/val/", 64), callbacks=[mc])

	model.save("../Models/cnn_only_model.h5")


# fine tune
# for layer in model.layers[:249]:
#    layer.trainable = False
# for layer in model.layers[249:]:
#    layer.trainable = True





