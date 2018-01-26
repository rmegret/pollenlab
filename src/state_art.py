from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.optimizers import SGD, Adam
import numpy as np
import os
from keras.models import Model
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Dense, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils.np_utils import to_categorical
from sklearn.cross_validation import train_test_split

def vgg_16(input_shape=(3,150,200),lr=0.000525):

	base_model = VGG16(weights='imagenet',include_top=False, input_shape=input_shape)
	pop_model = pop(base_model)
	x = pop_model.output
	x = Flatten(name='flatten')(x)
	predictions = Dense(2,activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	for layer in pop_model.layers:
		layer.trainable = False
	model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr),metrics=['accuracy'])

	return model 

def vgg_19(input_shape=(3,150,200),lr=0.000525):

	base_model = VGG19(weights='imagenet',include_top=False, input_shape=input_shape)
	pop_model = pop(base_model)
	x = pop_model.output
	x = Flatten(name='flatten')(x)
	predictions = Dense(2,activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	for layer in pop_model.layers:
		layer.trainable = False
	model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr),metrics=['accuracy'])
	return model 

def resnet_50(input_shape=(3,200,200),lr=0.000525):
	base_model = ResNet50(weights='imagenet',include_top=False, input_shape=input_shape)
	pop_model = pop(base_model)
	x = pop_model.output
	x = Flatten(name='flatten')(x)
	predictions = Dense(2,activation='softmax')(x)
	model = Model(inputs=base_model.input, outputs=predictions)
	for layer in pop_model.layers:
		layer.trainable = False
	model.compile(loss='binary_crossentropy', optimizer=SGD(lr=lr),metrics=['accuracy'])
	return model 

def pop(model):
	if not model.outputs:
		raise Exception('Sequential model cannot be popped: model is empty.')

	else:

		model.layers.pop()
		if not model.layers:
			model.outputs=[]
			model.inbound_nodes = []
			model.outbound_nodes = []
		else:
			model.layers[-1].outbound_nodes =[]
			model.outputs = [model.layers[-1].output]
		model.built = False
	return model 