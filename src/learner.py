from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D
from keras.layers.convolutional import Conv2D
from sklearn.preprocessing import scale

def logistic_regresor(units=1,input_dim=32*32*3,
						 activation='sigmoid', loss='binary_crossentropy',
						 				optimizer='sgd',metrics = 'accuracy'):
	model = Sequential()
	model.add(Dense(units, input_dim=input_dim, activation=activation))

	model.compile(loss=loss,
              optimizer=optimizer,
              metrics=[metrics])
	return model 


def ccd_layer( input_shape=(32, 32, 3), lr=0.0025):

	model = Sequential()

	model.add(Conv2D(16, (3,3), input_shape=input_shape ,name='1conv'))
	model.add(Activation('relu',name="1act"))
	model.add(MaxPooling2D(pool_size=(2,2),name="1pool"))

	model.add(Conv2D(16, (3,3),name="2conv"))
	model.add(Activation('relu',name="2act"))
	model.add(MaxPooling2D(pool_size=(2,2),name="2pool"))

	model.add(Flatten())
	model.add(Dense(16, activation='relu'))
	model.add(Dense(2, activation='softmax',name="4"))
	model.add(Dense(1))
	model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=lr),
              metrics=['accuracy'])

	return model

def ccd_layer_cat( input_shape=(32, 32, 3), lr=0.0025):

	model = Sequential()

	model.add(Conv2D(32, (3,3), input_shape=input_shape ,name='1conv'))
	model.add(Activation('relu',name="1act"))
	model.add(MaxPooling2D(pool_size=(2,2),name="1pool"))

	model.add(Conv2D(64, (3,3), input_shape=(32, 32, 32),name="2conv"))
	model.add(Activation('relu',name="2act"))
	model.add(MaxPooling2D(pool_size=(2,2),name="2pool"))

	model.add(Flatten())
	model.add(Dense(64, activation='relu'))
	model.add(Dense(2, activation='softmax',name="4"))
	model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=lr),
              metrics=['accuracy'])

	return model 

def shallow_model(input_shape=(3,100,100), lr =0.001, kernels=16, stride=(5,5),pool_size=(2,2), dense=50):
	model = Sequential()

	model.add(Conv2D(kernels, stride, input_shape=input_shape))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pool_size))

	model.add(Flatten())
	model.add(Dense(dense, activation='relu'))
	model.add(Dense(2, activation='softmax'))
	model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=lr),
              metrics=['accuracy'])
	return model

def squeeze():

	return 

def Liqi (input_shape=(3,100,100),kernels =[32,16,8,8,4],strides=[(5,5),(3,3),(3,3),(3,3),(1,1)],pools=[(2,2),(2,2)]):

	model = Sequential()

	model.add(Conv2D(kernels[0], strides[0], input_shape=input_shape,data_format="channels_first"))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pools[0]))
	model.add(BatchNormalization(axis=1))


	model.add(Conv2D(kernels[1], strides[1]))
	model.add(Activation('relu'))
	model.add(BatchNormalization(axis=1))

	model.add(Conv2D(kernels[2], strides[2]))	
	model.add(Activation('relu'))
	model.add(Conv2D(kernels[3], strides[3]))
	model.add(Activation('relu'))

	model.add(Conv2D(kernels[4], strides[4]))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=pools[1]))

	model.add(Flatten())
	model.add(Dense(2, activation='softmax',name="4"))
	model.compile(loss='binary_crossentropy',
              optimizer=SGD(lr=0.001),
              metrics=['accuracy'])
	return

