import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.layers import Input,Dropout,BatchNormalization,Activation,Add,Flatten,Dense,Reshape
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, UpSampling2D, Input, Cropping2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.mixed_precision import experimental as mixed_precision
import time
import json
import pandas as pd
import gc

# DEFINING FUNCTIONS
def plot_loss(metrics, figsave_dir):
	max_loss = max(metrics['loss'].values())
	max_valloss = max(metrics['val_loss'].values())
	# summarize metrics for loss
	plt.plot(list(metrics['loss'].values()))
	plt.plot(list(metrics['val_loss'].values()))
	plt.title('model loss')
	plt.ylabel('loss')
	plt.xlabel('epoch')
	plt.xlim(0, len(metrics['loss']))
	plt.ylim(0, max(max_loss, max_valloss))
	plt.legend(['train', 'val'], loc='upper right')
	plt.savefig(figsave_dir)
	#plt.show()


def NN_3():
	def down_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
		c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
		c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
		p = MaxPooling2D((2, 2), (2, 2))(c)
		return c, p

	def up_block(x, skip, filters, kernel_size=(3, 3), padding="same", strides=1):
		us = Conv2DTranspose(filters, kernel_size, strides = (2, 2), padding = 'same', activation='relu')(x)
		concat = Concatenate()([us, skip])
		c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(concat)
		c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
		return c

	def bottleneck(x, filters, kernel_size=(3, 3), padding="same", strides=1):
		c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(x)
		c = Conv2D(filters, kernel_size, padding=padding, strides=strides, activation="relu")(c)
		return c
		
	f = [8, 16, 32, 64, 128, 256, 512]
	inputs = Input((sampFreq, nreceivers, nshots))
	
	p0 = inputs
	c1, p1 = down_block(p0, f[0]) #128 -> 64
	c2, p2 = down_block(p1, f[1]) #64 -> 32
	c3, p3 = down_block(p2, f[2]) #32 -> 16
	c4, p4 = down_block(p3, f[3]) #16->8
	c5, p5 = down_block(p4, f[4])
	c6, p6 = down_block(p5, f[5])
	
	bn = bottleneck(p6, f[6])
	
	u1 = up_block(bn, c6, f[5]) #16 -> 32
	u2 = up_block(u1, c5, f[4]) #32 -> 64
	u3 = up_block(u2, c4, f[3]) #64 -> 128
	u4 = up_block(u3, c3, f[2])
	u5 = up_block(u4, c2, f[1])
	u6 = up_block(u5, c1, f[0])
	
	u7 = Conv2D(4, (3,3), padding='same', strides=(2, 1), activation="relu")(u6)
	
	outputs = Conv2D(1, (1, 1), padding="same", activation="relu", dtype='float32', name='predictions')(u7)
	
	model = Model(inputs, outputs)
	return model

#Global variables
sampFreq = 512
nreceivers = 256
nshots = 5
INPUT_SHAPE = (sampFreq, nreceivers, nshots)
#Train Variables
epochs = 200
batch_size = 16

if __name__ == '__main__':
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
	# print(tf.__version__)
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)
	#gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8)
	#sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
	
	dataset_type = 'SEGSALT' # Enter DUTCH or SEGSALT

	if dataset_type == 'DUTCH':
		for i in range(1, 5):
			gc.collect()
			#Loading dataset
			X_train = np.load('D:/WORK/SVMB/train_test/DI_DC/train/traces_set{}.npz'.format(i))
			X_train = X_train['data']
			y_train = np.load('D:/WORK/SVMB/train_test/DI_DC/train/tiles_set{}.npz'.format(i))
			y_train = y_train['data']
			y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)

			if i == 1:
				print("Initializing a fresh model.....")
				policy = mixed_precision.Policy('mixed_float16')
				mixed_precision.set_policy(policy)
				model = NN_3()
				model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
				print("Initializing Done!!!")
			else:
				print("Loading model......")
				policy = mixed_precision.Policy('mixed_float16')
				mixed_precision.set_policy(policy)
				model = load_model("D:/WORK/SVMB/trained_nets/DI_DC/NN3_set{}_16bit.h5".format(i-1))	
				print("Loading complete!!!")

			# DEFINE CALLBACKS
			Reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, verbose = 1, 
				mode = "min", min_delta = 1e-6, cooldown =5, min_lr = 1e-6)	
			Checkpoint = ModelCheckpoint(filepath='D:/WORK/SVMB/trained_nets/DI_DC/checkpoints/set{}_checkpoint.h5'.format(i),
				save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True)
			Tensorboard_callback = TensorBoard(log_dir="D:/WORK/SVMB/trained_nets/DI_DC/tensorboard/set{}/".format(i), 
				update_freq=70, histogram_freq=1, write_images=True)
			ES_callback = EarlyStopping(patience=7)

			#Fitting
			start_time = time.time()
			history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, 
				callbacks=[Checkpoint, Reduce, ES_callback], validation_split=0.1, shuffle=True)
			train_time = time.time() - start_time

			hist_df = pd.DataFrame(history.history)
			#SAVING METRICS
			hist_json_file = "D:/WORK/SVMB/trained_nets/DI_DC/metrics/history_set{}.json".format(i)
			with open(hist_json_file, mode='w') as f:
				hist_df.to_json(f)

			with open("D:/WORK/SVMB/train_test_Time.txt", "a") as text_file:
				print("Time taken to train {} for set{}: {} seconds\n".format(dataset_type, i, train_time), file=text_file)	

			#SAVING MODELS
			model.save('D:/WORK/SVMB/trained_nets/DI_DC/NN3_set{}_16bit.h5'.format(i))

			#DISPLAY AND SAVE LOSS PLOT
			with open('D:/WORK/SVMB/trained_nets/DI_DC/metrics/history_set{}.json'.format(i)) as f:
				metrics = json.load(f)
			plot_loss(hist_df, figsave_dir='D:/WORK/SVMB/trained_nets/DI_DC/images/NN3_set{}_16bit.png'.format(i))

			del X_train, y_train
			

	elif dataset_type == 'SEGSALT':
		train_type = 'fine_tuning' #Enter fine_tuning or fresh_training
		lst = [304, 608, 1014]
		for item in lst:
			gc.collect()
			#Loading dataset
			train_data = np.load('D:/WORK/SVMB/train_test/segsalt/train/set{}_train.npz'.format(item))
			X_train = train_data['tr']
			y_train = train_data['ti']
			y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], y_train.shape[2], 1)

			if train_type == 'fresh_training':
				print("Initializing a fresh model.....")
				policy = mixed_precision.Policy('mixed_float16')
				mixed_precision.set_policy(policy)
				model = NN_3()
				model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mse'])
				print("Initializing Done!!!")
			elif train_type == 'fine_tuning':
				print("Loading model......")
				policy = mixed_precision.Policy('mixed_float16')
				mixed_precision.set_policy(policy)
				model = load_model("D:/WORK/SVMB/trained_nets/DI_DC/NN3_set{}_16bit.h5".format(4))	
				print("Loading complete!!!")
			else:
				print("INVALID train_type!!!")	

			# DEFINE CALLBACKS
			Reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=4, verbose = 1, 
				mode = "min", min_delta = 1e-6, cooldown =5, min_lr = 1e-6)	
			Checkpoint = ModelCheckpoint(filepath='D:/WORK/SVMB/trained_nets/SEGSALT/{}/checkpoints/set{}_checkpoint.h5'.format(train_type, item),
				save_weights_only=True,monitor='val_loss',mode='min',save_best_only=True)
			Tensorboard_callback = TensorBoard(log_dir="D:/WORK/SVMB/trained_nets/SEGSALT/{}/tensorboard/set{}/".format(train_type, item),
				update_freq=70, histogram_freq=1, write_images=True)
			ES_callback = EarlyStopping(patience=7)

			#Fitting
			start_time = time.time()
			history = model.fit(x=X_train, y=y_train, batch_size=batch_size, epochs=epochs, 
				callbacks=[Checkpoint, Reduce, ES_callback], validation_split=0.1, shuffle=True)
			train_time = time.time() - start_time

			hist_df = pd.DataFrame(history.history)
			#SAVING METRICS
			hist_json_file = "D:/WORK/SVMB/trained_nets/SEGSALT/{}/metrics/history_set{}.json".format(train_type, item)
			with open(hist_json_file, mode='w') as f:
				hist_df.to_json(f)

			with open("D:/WORK/SVMB/train_test_Time.txt", "a") as text_file:
				print("Time taken to train {} for {} {}: {} seconds\n".format(dataset_type, item, train_type, train_time), file=text_file)	

			#SAVING MODELS
			model.save('D:/WORK/SVMB/trained_nets/SEGSALT/{}/NN3_set{}_16bit.h5'.format(train_type, item))

			#DISPLAY AND SAVE LOSS PLOT
			with open('D:/WORK/SVMB/trained_nets/SEGSALT/{}/metrics/history_set{}.json'.format(train_type, item)) as f:
				metrics = json.load(f)
				plot_loss(metrics, figsave_dir='D:/WORK/SVMB/trained_nets/SEGSALT/{}/images/NN3_set{}_16bit.png'.format(train_type, item))

			del X_train, y_train

	else:
		print("INVALID dataset_type!!!!!")		

		

