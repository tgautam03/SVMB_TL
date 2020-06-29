import os
import sys
import random
import numpy as np
import matplotlib as m
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input,Dropout,BatchNormalization,Activation,Add,Flatten,Dense,Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D, Input, Cropping2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
import time
import pandas as pd
import gc

def plot1x2Array(image, mask1, v_min, v_max, k, plot_save_dict):
	#invoke matplotlib!
	f, ax = plt.subplots(1,2)
	im1 = ax[0].imshow(image, vmin=v_min, vmax=v_max, aspect='auto')
	im = ax[1].imshow(mask1, vmin=v_min, vmax=v_max, aspect='auto')
	f.colorbar(im1)

	ax[0].grid()
	ax[1].grid()

	ax[0].set_title('Model number {}'.format(k))
	ax[1].set_title('Predicted Model')

	plt.savefig(plot_save_dict + str(k) + '.png')
	#plt.show()

def vellog_1x2Array(image, mask1, k, plot_save_dict):
	Y = np.linspace(0, 256, num=256)
	Y=Y.reshape(Y.shape[0],1)
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

	vel11 = image[:, 49] 
	vel12 = image[:, 124]
	vel13 = image[:, 199]

	vel21 = mask1[:, 49] 
	vel22 = mask1[:, 124]
	vel23 = mask1[:, 199]

	f, ax = plt.subplots(1,3)
	ax[0].plot(vel11[:],Y[:], "-b", label="Ground Truth")
	ax[0].plot(vel21[:],Y[:], "-r", label="Predicted")
	ax[0].set_ylim([256,0])
	ax[0].legend(loc="upper right")

	ax[1].plot(vel12[:],Y[:], "-b", label="Ground Truth")
	ax[1].plot(vel22[:],Y[:], "-r", label="Predicted")
	ax[1].set_ylim([256,0])
	ax[1].legend(loc="upper right")

	ax[2].plot(vel13[:],Y[:], "-b", label="Ground Truth")
	ax[2].plot(vel23[:],Y[:], "-r", label="Predicted")
	ax[2].set_ylim([256,0])
	ax[2].legend(loc="upper right")

	ax[0].grid()
	ax[1].grid()
	ax[2].grid()

	ax[0].set_title('Log at 500 m from left (49 pixel mark)')
	ax[1].set_title('Log at 1.25 km from left (124 pixel mark)')
	ax[2].set_title('Log at 2 km from left (199 pixel mark)')

	plt.savefig(plot_save_dict + str(k) + '_vellogs.png')
	#plt.show()    


def plot1x3Array(image, mask1, mask2, v_min, v_max, k, plot_save_dict):
	#invoke matplotlib!
	f, ax = plt.subplots(1,3)
	im1 = ax[0].imshow(image, vmin=v_min, vmax=v_max, aspect='auto')
	im = ax[1].imshow(mask1, vmin=v_min, vmax=v_max, aspect='auto')
	im = ax[2].imshow(mask2, vmin=v_min, vmax=v_max, aspect='auto')
	f.colorbar(im1)

	ax[0].grid()
	ax[1].grid()
	ax[2].grid()

	ax[0].set_title('Model number {}'.format(k))
	ax[1].set_title('Predicted from CNN fresh_training')
	ax[2].set_title('Predicted from CNN fine_tuning')

	plt.savefig(plot_save_dict + str(k) + '_models.png')
	#plt.show()

def vellog_1x3Array(image, mask1, mask2, k, plot_save_dict):
	Y = np.linspace(0, 256, num=256)
	Y=Y.reshape(Y.shape[0],1)
	plt.rcParams['xtick.bottom'] = plt.rcParams['xtick.labelbottom'] = False
	plt.rcParams['xtick.top'] = plt.rcParams['xtick.labeltop'] = True

	vel11 = image[:, 49] 
	vel12 = image[:, 124]
	vel13 = image[:, 199]

	vel21 = mask1[:, 49] 
	vel22 = mask1[:, 124]
	vel23 = mask1[:, 199]

	vel31 = mask2[:, 49] 
	vel32 = mask2[:, 124]
	vel33 = mask2[:, 199]

	f, ax = plt.subplots(1,3)
	ax[0].plot(vel11[:],Y[:], "-b", label="Ground Truth")
	ax[0].plot(vel21[:],Y[:], "-r", label="Predicted from CNN fresh_training")
	ax[0].plot(vel31[:],Y[:], "-g", label="Predicted from CNN fine_tuning")
	ax[0].set_ylim([256,0])
	ax[0].legend(loc="upper right")

	ax[1].plot(vel12[:],Y[:], "-b", label="Ground Truth")
	ax[1].plot(vel22[:],Y[:], "-r", label="Predicted from CNN fresh_training")
	ax[1].plot(vel32[:],Y[:], "-g", label="Predicted from CNN fine_tuning")
	ax[1].set_ylim([256,0])
	ax[1].legend(loc="upper right")

	ax[2].plot(vel13[:],Y[:], "-b", label="Ground Truth")
	ax[2].plot(vel23[:],Y[:], "-r", label="Predicted from CNN fresh_training")
	ax[2].plot(vel33[:],Y[:], "-g", label="Predicted from CNN fine_tuning")
	ax[2].set_ylim([256,0])
	ax[2].legend(loc="upper right")

	ax[0].grid()
	ax[1].grid()
	ax[2].grid()

	ax[0].set_title('Log at 500 m from left (49 pixel mark)')
	ax[1].set_title('Log at 1.25 km from left (124 pixel mark)')
	ax[2].set_title('Log at 2 km from left (199 pixel mark)')

	plt.savefig(plot_save_dict + str(k) + '_vellogs.png')
	#plt.show()

plt.rcParams['figure.figsize'] = [20, 5]   

if __name__ == '__main__':
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	# print(tf.__version__)
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
	#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
	
	dataset_type = 'SEGSALT' # Enter DUTCH or SEGSALT

	if dataset_type == 'DUTCH':
		set_num = 4 # Select set_num from 1 to 4
		#Loading dataset
		X_test = np.load('D:/WORK/SVMB/train_test/DI_DC/test/traces.npz')
		X_test = X_test['data']
		y_test = np.load('D:/WORK/SVMB/train_test/DI_DC/test/tiles.npz')
		y_test = y_test['data']
		y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

		model = load_model("D:/WORK/SVMB/trained_nets/DI_DC/NN3_set{}_16bit.h5".format(set_num))

		start_time = time.time()
		y_pred = model.predict(X_test, batch_size=16)
		train_time = time.time() - start_time
		metrics = model.evaluate(X_test, y_test, batch_size=16)
		with open("D:/WORK/SVMB/train_test_Time.txt", "a") as text_file:
				print("Time taken to predict {} for set_num{}: {} seconds\n Evaluation metrics :{}".format(dataset_type, set_num, train_time, metrics), file=text_file)
		print("Test MSE : ", metrics)
		metrics_df = pd.DataFrame(metrics)
		metrics_json_file = "D:/WORK/SVMB/test_results/DI_DC/set{}/metrics.json".format(set_num)
		with open(metrics_json_file, mode='w') as f:
			metrics_df.to_json(f)

		for i in range(25):
			k = np.random.randint(X_test.shape[0])
			image = y_test[k, :, :, 0]
			mask1 = y_pred[k, :, :, 0]
			plot1x2Array(image, mask1, v_min = np.min(image), v_max = np.max(image), k=k, 
				plot_save_dict="D:/WORK/SVMB/test_results/DI_DC/set{}/".format(set_num))
			vellog_1x2Array(image, mask1, k=k, plot_save_dict="D:/WORK/SVMB/test_results/DI_DC/set{}/".format(set_num))

	elif dataset_type == 'SEGSALT':
		allPred = False # True if DI_DC trained network is used to test all segsalt models

		if allPred == True:
			set_num = 4
			#Loading dataset
			X_test = np.load('D:/WORK/SVMB/segSaltData/traces.npz')
			X_test = X_test['data']
			y_test = np.load('D:/WORK/SVMB/segSaltData/tiles.npz')
			y_test = y_test['data']
			y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

			model = load_model("D:/WORK/SVMB/trained_nets/DI_DC/NN3_set{}_16bit.h5".format(set_num))

			start_time = time.time()
			y_pred = model.predict(X_test, batch_size=16)
			train_time = time.time() - start_time
			metrics = model.evaluate(X_test, y_test, batch_size=16)
			with open("D:/WORK/SVMB/train_test_Time.txt", "a") as text_file:
					print("Time taken to predict {} using network trained on DI_DC set_num{}: {} seconds\n Evaluation metrics :{}".format(dataset_type, set_num, train_time, metrics), file=text_file)
			print("Test MSE : ", metrics)
			metrics_df = pd.DataFrame(metrics)
			metrics_json_file = "D:/WORK/SVMB/test_results/segsalt/allPred/metrics.json"
			with open(metrics_json_file, mode='w') as f:
				metrics_df.to_json(f)

			for i in range(25):
				k = np.random.randint(X_test.shape[0])
				image = y_test[k, :, :, 0]
				mask1 = y_pred[k, :, :, 0]
				plot1x2Array(image, mask1, v_min = np.min(image), v_max = np.max(image), k=k, 
					plot_save_dict="D:/WORK/SVMB/test_results/segsalt/allPred/")
				vellog_1x2Array(image, mask1, k=k, plot_save_dict="D:/WORK/SVMB/test_results/segsalt/allPred/")

		else:
			lst = [304, 608, 1014]
			for item in lst:
				gc.collect()
				#Loading dataset
				test_data = np.load('D:/WORK/SVMB/train_test/segsalt/test/set{}_test.npz'.format(item))
				X_test = test_data['tr']
				y_test = test_data['ti']
				y_test = y_test.reshape(y_test.shape[0], y_test.shape[1], y_test.shape[2], 1)

				model0 = load_model("D:/WORK/SVMB/trained_nets/segsalt/fine_tuning/NN3_set{}_16bit.h5".format(item))
				model1 = load_model("D:/WORK/SVMB/trained_nets/segsalt/fresh_training/NN3_set{}_16bit.h5".format(item))

				start_time = time.time()
				y_pred0 = model0.predict(X_test, batch_size=16)
				y_pred1 = model1.predict(X_test, batch_size=16)
				train_time = time.time() - start_time

				metrics0 = model0.evaluate(X_test, y_test, batch_size=16)
				metrics1 = model1.evaluate(X_test, y_test, batch_size=16)
				with open("D:/WORK/SVMB/train_test_Time.txt", "a") as text_file:
					print("Time taken to predict fine_tuning and fresh_training {} for set_num{}: {} seconds\n Evaluation metrics fine_tuning:{}\n Evaluation metrics fresh_training:{}\n".format(dataset_type, item, train_time, metrics0, metrics1), file=text_file)
				print("Test MSE fine_tuning: ", metrics0)
				print("Test MSE fresh_training: ", metrics1)

				metrics0_df = pd.DataFrame(metrics0)
				metrics0_json_file = "D:/WORK/SVMB/test_results/segsalt/set{}/metrics_fine_tuning.json".format(item)
				with open(metrics0_json_file, mode='w') as f:
					metrics0_df.to_json(f)
				metrics1_df = pd.DataFrame(metrics1)
				metrics1_json_file = "D:/WORK/SVMB/test_results/segsalt/set{}/metrics_fresh_training.json".format(item)
				with open(metrics1_json_file, mode='w') as f:
					metrics1_df.to_json(f)	

				for i in range(25):
					k = np.random.randint(X_test.shape[0])
					image = y_test[k, :, :, 0]
					mask1 = y_pred0[k, :, :, 0]
					mask2 = y_pred1[k, :, :, 0]
					plot1x3Array(image, mask2, mask1, v_min = np.min(image), v_max = np.max(image), k=k, 
						plot_save_dict='D:/WORK/SVMB/test_results/segsalt/set{}/'.format(item))
					vellog_1x3Array(image, mask2, mask1, k=k, 
						plot_save_dict='D:/WORK/SVMB/test_results/segsalt/set{}/'.format(item))	
