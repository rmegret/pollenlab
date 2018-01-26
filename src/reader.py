import glob
import os
import cv2 as cv
import numpy as np
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from .normalizer import Scaler

def dataset(file_list,size,flattened=False):
	data = []
	for i, file in enumerate(file_list):
		image = cv.imread(file)
		image2 = cv.cvtColor(image, cv.COLOR_RGB2BGR)
		image = cv.resize(image2, size)
		if flattened:
			image = image.flatten()

		data.append(image)

	labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]

	return np.array(data), np.array(labels)


def norm_dataset(file_list,size,flattened=False):
	data = []
	for i, file in enumerate(file_list):
		image = cv.imread(file)
		image2 = cv.cvtColor(image, cv.COLOR_RGB2BGR)
		image = cv.resize(image2, size)
		image = cv.normalize(image,dst=image, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
		if flattened:
			image = image.flatten()

		data.append(image)

	return np.array(data)

def white_dataset(file_list, size):

	data = []
	for i, file in enumerate(file_list):
		image = np.loadtxt(file)
		#img = np.array([image,image,image])
		data.append(image)

	labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]
	return np.array(data), np.array(labels)



class Dataset_builder(Scaler):

	

	def __init__(self, path, size=(32, 32)):
		self.path = path
		self.size = size
		self.file_list = glob.glob(os.path.join(self.path, '*.jpg'))
		self.data, self.labels = dataset(self.file_list,self.size)
		self.normalized_data = norm_dataset(self.file_list, self.size)
		self.data_flattened= dataset(self.file_list,self.size,True)
		self.labels_categorical = to_categorical(self.labels)
		Scaler.__init__(self,self.data,size)
		

	def white_dataset(self,ext="ye"):

		data = []
		for i, file in enumerate(self.file_list):

			image = np.loadtxt(file[:-3]+ext)
			img = np.array([image,image,image])
			data.append(img)

		labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in self.file_list]

		self.data,self.labels = np.array(data), np.array(labels)

		return 

	def split_in_folders(self, test_size=0.25):
		
		for i, file in enumerate(self.file_list):
			image = cv.imread(file)
			parts = round(1 / test_size)
			if i % parts == 0:
				if file[13:14] == "P":
					os.makedirs(os.path.dirname(file[:13]+ "test/P/"), exist_ok=True)
					cv.imwrite(file[:13] + "test/P/" + file[13:], image)
				else:
					os.makedirs(os.path.dirname(file[:13]+ "test/N/"), exist_ok=True)
					cv.imwrite(file[:13] + "test/N/" + file[13:], image)
			else:
				if file[13:14] == "P":
					os.makedirs(os.path.dirname(file[:13]+ "train/P/"), exist_ok=True)
					cv.imwrite(file[:13] + "train/P/" + file[13:], image)
				else:
					os.makedirs(os.path.dirname(file[:13]+ "train/N/"), exist_ok=True)
					cv.imwrite(file[:13] + "train/N/" + file[13:], image)
		return

	def splitnoshuffle(self,test_size=0.25,scaled=False):
		
		if scaled:
			try:
				data= self.scaled_data
			except: 
				print("The data has not been scaled")
				return 
		else:
			data = self.normalized_data

		print("spliting in stratified no shuffling way....")
		# Takes the positions of the classes
		id0=np.where(self.labels==0)
		id1=np.where(self.labels==1)

		
		N0=len(id0[0]) # Number of elements with class 0 
		n0=round(N0*test_size) # Amount of data to take into testing 
	
		X0=data[id0[0],:] # taking the elements with class 0
		X0test=X0[range(n0),:] # taking the first n0 elements of class 0 
		X0train=X0[range(n0,N0-n0),:] # taking the rest as training data
	
		y0test=np.zeros((len(X0test),1),dtype = int)
		y0train = np.zeros((len(X0train),1),dtype =int)
	
		N1=len(id1[0])
		n1=round(N1*test_size)
	
		X1=data[id1[0],:]
		X1test=X1[range(n1),:]
		X1train=X1[range(n1,N1-n1),:]
		y1test= np.ones((len(X1test),1))
		y1train = np.ones((len(X1train),1))
	
   
		X_train = list(X0train)+list(X1train)
		X_test = list(X0test)+list(X1test)  
		y_train = list(y0train)+list(y1train)
		y_test = list(y0test)+list(y1test) 

		return np.array(X_train),np.array(X_test),np.array(y_train,dtype=int),np.array(y_test,dtype=int)

	def split_train_test(self,test_size = 0.25,scaled=False,whiteness=False):
		if scaled:
			try:
				data= self.scaled_data
			except: 
				print("The data has not been scaled")
				return 
		elif whiteness:

			try: 
				data= self.data 
			except: 
				print("check data for split and train")
				return 
		else: 
			data = self.normalized_data
		X_train, X_test, y_train, y_test = train_test_split(data, self.labels, test_size=.25,random_state=0)

		return np.array(X_train),np.array(X_test),np.array(y_train,dtype=int),np.array(y_test,dtype=int)
	 






		
