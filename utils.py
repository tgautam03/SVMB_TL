import numpy as np
import matplotlib.pyplot as plt
import os
from examples.seismic import Model, plot_velocity, AcquisitionGeometry, plot_shotrecord, Receiver
from examples.seismic.acoustic import AcousticWaveSolver
from devito import Function, clear_cache, TimeFunction
from matplotlib import cm
import scipy.io
from sklearn.model_selection import train_test_split

class Display:
	def __init__(self, file):
		self.file = file

	def dispVelMods(self):
		fig=plt.figure(figsize=(50, 50))	
		columns, rows = 4, 4	
		for i in range(1, columns*rows +1):
			k = np.random.randint(self.file.shape[0])
			img = self.file[k]
			fig.add_subplot(rows, columns, i)
			im = plt.imshow(img)
			plt.colorbar(im)
		plt.show()
			


class FDM:
	def __init__(self, file, shape, spacing, origin, nbl, t0, tn, f0, 
		sampFreq, nshots, nreceivers, shot_number, dispTrace):
		self.file = file
		self.shape = shape
		self.spacing = spacing
		self.origin = origin
		self.nbl = nbl
		self.t0 = t0
		self.tn = tn
		self.f0 = f0
		self.sampFreq = sampFreq
		self.nshots = nshots
		self.nreceivers = nreceivers
		self.shot_number = shot_number
		self.dispTrace = dispTrace
		
		
	def forward(self):
		v = self.file
		vmod = v[0].T
		directory = "/home/tushar/work/SVMB/final_run/seg_salt/segSalt_traces/"
		spacing = self.spacing
		model = Model(vp=vmod, origin=self.origin, shape=self.shape, spacing=spacing, 
			space_order=2, nbl=self.nbl)
		model0 = Model(vp=vmod[0,0]*np.ones(self.shape, dtype=np.float32), origin=self.origin, 
			shape=self.shape, spacing=self.spacing, space_order=2, nbl=self.nbl)

		t0=self.t0
		tn=self.tn
		f0=self.f0

		nreceivers = self.nreceivers
		rec_coordinates = np.empty((nreceivers, 2))
		rec_coordinates[:, 0] = np.linspace(0, model.domain_size[0], num=nreceivers)
		rec_coordinates[:, 1] = 20.

		nshots = self.nshots
		shot_id = np.linspace(spacing[0], model.domain_size[0] - spacing[0], num=nshots)
		src_coordinates = np.empty((1, 2))
		src_coordinates[0, :] = shot_id[self.shot_number]
		src_coordinates[0, -1] = 20.  # Depth is 20m

		geometry = AcquisitionGeometry(model, rec_coordinates, src_coordinates, t0, tn, 
			f0=f0, src_type='Ricker')
		solver = AcousticWaveSolver(model, geometry, space_order=2)

		#SHOT 
		true_d, _, _ = solver.forward(vp=model.vp)
		trace = true_d.resample(num=self.sampFreq)

		#DIRECT
		true_d, _, _ = solver.forward(vp=model0.vp)
		direct = true_d.resample(num=self.sampFreq)

		# Final trace
		tr = trace.data-direct.data

		if self.dispTrace == True:
			fig=plt.figure(figsize=(50, 50))		
			plot_velocity(model)
			plot_shotrecord(tr.data, model, t0, tn)
			plt.show()
		
		np.savez_compressed(directory + "m" + str(0) + "_shot" + str(self.shot_number) + ".npz", trace=tr.data)

		for i in range(1, v.shape[0]):
			# Define a velocity profile. The velocity is in km/s
			vmod = v[i].T 
			model = Model(vp=vmod, origin=self.origin, shape=self.shape, spacing=spacing, 
				space_order=2, nbl=self.nbl)
			model0 = Model(vp=vmod[0,0]*np.ones(self.shape, dtype=np.float32), origin=self.origin, 
				shape=self.shape, spacing=self.spacing, space_order=2, nbl=self.nbl)
			#SHOT 
			true_d, _, _ = solver.forward(vp=model.vp)
			trace = true_d.resample(num=512)

			#DIRECT
			true_d, _, _ = solver.forward(vp=model0.vp)
			direct = true_d.resample(num=512)

			# Final trace
			tr = trace.data-direct.data

			np.savez_compressed(directory + "m" + str(i) + "_shot" + str(self.shot_number) + ".npz", trace=tr.data)	


def concatTraces(file, directory, start_models, end_models, sampFreq, nreceivers, nshots):

	num_models = end_models - start_models
	traces_set = np.empty((num_models, sampFreq, nreceivers, nshots), dtype=np.float32)

	tiles_set = file[start_models:end_models]
	
	for i in range(start_models, end_models):
		for j in range(nshots):
			tr = np.load(directory + "/m{}_shot{}.npz".format(i, j))
			traces_set[i, :, :, j] = tr["trace"]

	return tiles_set, traces_set		

def trainTestSplit(tr, ti, num_test):
	tr_train, ti_train, tr_test, ti_test = train_test_split(tr, ti, test_size=num_test)
	return tr_train, ti_train, tr_test, ti_test




