from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D
from keras.layers.convolutional import Conv2D
from sklearn.preprocessing import scale
from keras import regularizers
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from src.util import dataset 
from src.model import *
from src.evaluator import Experiment
from keras.utils.np_utils import to_categorical



def main():
	path="../pollendataset/Dataset/"
    imlist= glob.glob(os.path.join(path, '*.jpg'))
	data,labels= dataset(imlist)
    X_train, X_test, y_train, y_test = train_test_split(data,labels, test_size=.25,random_state=0)
	
	for k in [4,8]:
		for s in [(1,1),(3,3),(7,7),(15,15)]:
			for p in [(2,2),(4,4),(8,8)]:
				for d in [5,10,20]:
					for lr in [0.0005]:
						model= two_layer_model(lr =lr, kernels=k, stride=s,pool_size=p, dense=d)
						experiment =Experiment(X,to_categorical(Y),X_t,to_categorical(Y_t))
						experiment.path = "/pylon5/ci5616p/piperod/pollenlab/results/Bridges_clean_two_layer"
						experiment.epochs = 200
						experiment.launch_experiment(model)


if __name__ == '__main__':
	main()
