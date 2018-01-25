from sklearn.metrics import accuracy_score, confusion_matrix
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.optimizers import SGD, Adam
from keras.utils.np_utils import to_categorical
from keras.layers.convolutional import Convolution2D, MaxPooling2D, Convolution3D
from keras.layers.convolutional import Conv2D
from sklearn.preprocessing import scale
from keras import regularizers
from vis.visualization import visualize_saliency, overlay,visualize_cam
from vis.utils import utils
from keras import activations
import numpy as np 
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import os
import matplotlib.pyplot as plt 
import keras 
from matplotlib.backends.backend_pdf import PdfPages
from keras.callbacks import LearningRateScheduler, ModelCheckpoint, CSVLogger, TensorBoard

class Experiment(object):

	def __init__(self,X,Y,X_t,Y_t):

		self.train = X 
		self.train_y = Y
		self.test = X_t
		self.test_y = Y_t
		self.path = "Results"
		self.optimizer = 'sgd'
		self.epochs = 20 
		self.batch_size = 100

	def launch_experiment(self,model,vis=False):

		try:
            
            print("...Launching Experiment....")
            os.makedirs(self.path,exist_ok=True)
            Nexp= len(os.listdir(self.path)) #How many directories are there in results
            os.makedirs(self.path+"/Experiment"+str(Nexp))
            self.path_to_save = self.path+"/Experiment"+str(Nexp)
            os.makedirs(self.path_to_save+'/logs',exist_ok=True)
            # Callbacks 
            WEIGHTS_BEST = self.path_to_save+'/weights.best.h5'
            TRAINING_LOG = self.path_to_save+'/logs/training_log,csv'
            LOGS_DIR = self.path_to_save+'./logs'
            checkpoint = ModelCheckpoint(WEIGHTS_BEST, monitor='loss',                              verbose=0, save_best_only=False,
                                 save_weights_only=True,
                                         mode='min', period=1) 
            csv_logger = CSVLogger(TRAINING_LOG, append=True)
            tb = TensorBoard(log_dir=LOGS_DIR, histogram_freq=0,                                            write_graph=True, write_images=False)
            callbacks_list = [ checkpoint, csv_logger, tb]
            # Compiling the model and saving it 
            history = model.fit(self.train,self.train_y,
                                batch_size =self.batch_size,
                                epochs=self.epochs,
                                validation_data=(self.test,self.test_y)
                                callbacks=calbacks_list)
            results = model.evaluate(self.test,self.test_y)
            H = pd.DataFrame(history.history, index=history.epoch)
		except:
			print("Check inputs and outputs dimensions")
			return

		self.model = model #writing the model in the object
		os.makedirs(self.path+"/Experiment"+str(Nexp))
		self.path_to_save = self.path+"/Experiment"+str(Nexp)
		print(".........The experiment was successfull.........")
		print("....Saving model to disk....")
		model_json = model.to_json() #save the model
		filename = self.path_to_save+"/model.json"
		with open(filename, "w") as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		model.save_weights(self.path_to_save+"/model.h5") #save last weights and all the model
		H.to_csv(self.path_to_save+"/results_per_epoch.csv", index=False, encoding='utf-8') #writing the history on a csv file
		#H.plot()
		#plt.title("Test accuracy: {:3.1f} %".format(results[-1]*100), fontsize=15)

		#plt.savefig(self.path_to_save + "/results.png")
		print("..model saved...")


		return 

	def launch_visualization(self,img1,img2):

		model = self.model #load the model 
		layers = model.layers 
		#retrieve layers with convolution  and with activation 
		convlayers = np.array([1 if type(layer) is keras.layers.convolutional.Conv2D else 0 for layer in layers])
		actlayers = np.array([1 if type(layer) is keras.layers.core.Activation else 0 for layer in layers])
		denselayers = np.array([1 if type(layer) is keras.layers.core.Activation else 0 for layer in layers])

		if convlayers.sum() == 0:
			print("There is no conv layers")
			return 
		vis_path=self.path_to_save+"/visualization"
		os.makedirs(vis_path,exist_ok=True)
		
		for i,j in enumerate(convlayers):
			if j==1 and i>=1:
				layer_name=layers[i].get_config()['name']
				filters = layers[i].filters
				idx=range(filters)
				nx = int(np.ceil(np.sqrt(filters)))+1
				ny = int(np.ceil(filters/nx))+1


				f, ax = plt.subplots(int(ny), int(nx)*2)

				ax[ny-1,nx-1].imshow(np.uint8(img1))
				ax[ny-1,2*nx-1].imshow(np.uint8(img2))

				layer_idx = utils.find_layer_idx(model,layer_name)
				for i, img in enumerate([img1, img2]): 
					for axj, j in enumerate(idx):
						#heatmap=np.zeros((img.shape))
						#heatmap = visualize_cam(model,layer_idx, filter_indices=j, 
								#seed_input=img.transpose(2,0,1))
						heatmap = visualize_saliency(model, layer_idx, filter_indices=j, seed_input=img.transpose(2,0,1))
						# Lets overlay the heatmap onto original image.    
						if i==0:
							ax[axj//nx,axj%nx].imshow(overlay(img, heatmap))
							#ax[(j)//4,(j)%4].set_title(str(j%7)+str(j%3))
						if i==1:
							ax[axj//nx,axj%nx+nx].imshow(overlay(img, heatmap))
			
				plt.savefig(vis_path+"/"+layer_name)



		for i,j in enumerate(actlayers):
			if j==1 and i>=2:
				layer_name=layers[i].get_config()['name']
				filters = layers[i].input_shape[1]
				idx=range(filters)
				nx = int(np.ceil(np.sqrt(filters)))+1
				ny = int(np.ceil(filters/nx))+1

				f, ax = plt.subplots(int(ny), int(nx)*2)

				ax[ny-1,nx-1].imshow(np.uint8(img1))
				ax[ny-1,2*nx-1].imshow(np.uint8(img2))

				layer_idx = utils.find_layer_idx(model,layer_name)
				for i, img in enumerate([img1, img2]): 
					for axj, j in enumerate(idx):
						#heatmap=np.zeros((img.shape))
						heatmap = visualize_cam(model,layer_idx, filter_indices=j, 
								seed_input=img.transpose(2,0,1))
						#heatmap = visualize_saliency(model, layer_idx, filter_indices=j, seed_input=img.transpose(2,0,1))
						# Lets overlay the heatmap onto original image.    
						if i==0:
							ax[axj//nx,axj%nx].imshow(overlay(img, heatmap))
							#ax[(j)//4,(j)%4].set_title(str(j%7)+str(j%3))
						if i==1:
							ax[axj//nx,axj%nx+nx].imshow(overlay(img, heatmap))
			
				plt.savefig(vis_path+"/"+layer_name)


		

		return

	def launch_contact(self,X_t,std,mean):

		model = self.model 

		predictions = model.predict(X_t)

		X = X_t.transpose(0,2,3,1)

		X = X*std*-1+mean

		idx=range(len(X))
		nx = 4
		ny = 6
		f, ax = plt.subplots(int(ny), int(nx))
		i=0
		contact=self.path_to_save+"/contact"
		with PdfPages(contact+'.pdf') as pdf:
			for j,x in enumerate(X):
				ax[i//nx,i%nx].imshow(x)
				ax[i//nx,i%nx].set_title("P = "+ str(predictions[j]))
				i+=1
				if i%24==0:
					i=0
					pdf.savefig()
					f, ax = plt.subplots(int(ny), int(nx))
			plt.savefig()
		return



