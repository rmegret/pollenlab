import numpy as np 
import pandas as pd
import json 
from keras.models import model_from_json

from skimage import io, transform
def dataset(file_list,size=(300,180),flattened=False):
	data = []
	for i, file in enumerate(file_list):
		image = io.imread(file)
		image = transform.resize(image, size, mode='constant')
		if flattened:
			image = image.flatten()

		data.append(image)

	labels = [1 if f.split("/")[-1][0] == 'P' else 0 for f in file_list]

	return np.array(data), np.array(labels)


def load_json(filename):
	with open(filename) as input_file:
		loaded= json.load(input_file)
	return loaded 

def load_model(json_file,weights_file):
	loaded_model = load_json(json_file)
	model=model_from_json(loaded_model)
	model.load_weights(weights_file)
	return model
