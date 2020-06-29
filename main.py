import numpy as np
import matplotlib.pyplot as plt
from utils import Display, FDM, concatTraces, trainTestSplit

vmodType = "INLINE"
concat = False
# directory = "/home/tushar/work/SVMB/final_run/final_Dutch{}.npy".format(vmodType)
directory = "/home/tushar/work/SVMB/final_run/seg_salt/seg_saltMods.npz"
file = np.load(directory)
if directory[-1] == 'z':
	file = file['data']

dispObj = Display(file)
dispObj.dispVelMods()

for i in range(2, 3):
	FDMobj = FDM(file=file, shape=(256,256), spacing=(10.,10.), origin=(0.,0.), nbl=100, t0=0, 
		tn=3000., f0=0.010, sampFreq=512, nshots=5, nreceivers=256, shot_number=i, 
		dispTrace=False)
	FDMobj.forward()	

if concat == True:
	tiles_set, traces_set = concatTraces(file, 0, 3000, 512, 256, 5)	
	np.savez_compressed("/home/tushar/work/SVMB/final_run/{}traces_set1.npz".format(vmodType), traces_set1=traces_set)
	np.savez_compressed("/home/tushar/work/SVMB/final_run/{}tiles_set1.npz", tiles_set1=tiles_set)

	tiles_set, traces_set = concatTraces(file, 3000, 6000, 512, 256, 5)	
	np.savez_compressed("/home/tushar/work/SVMB/final_run/{}traces_set2.npz".format(vmodType), traces_set2=traces_set)
	np.savez_compressed("/home/tushar/work/SVMB/final_run/{}tiles_set2.npz".format(vmodType), tiles_set2=tiles_set)

	tiles_set, traces_set = concatTraces(file, 6000, file.shape[0], 512, 256, 5)	
	np.savez_compressed("/home/tushar/work/SVMB/final_run/{}traces_set3.npz".format(vmodType), traces_set3=traces_set)
	np.savez_compressed("/home/tushar/work/SVMB/final_run/{}tiles_set3.npz".format(vmodType), tiles_set3=tiles_set)


directory="/home/tushar/work/SVMB/final_run/seg_salt/segSalt_traces/"
tiles_set, traces_set = concatTraces(file, directory, 0, 2028, 512, 256, 5)	
np.savez_compressed("/home/tushar/work/SVMB/final_run/seg_salt/segSalt_traces.npz", data=traces_set)

def saltSEG_trainTestSplit(test_nums):
	tr = np.load("/home/tushar/work/SVMB/final_run/seg_salt/segSalt_traces.npz")
	tr = tr['data']

	ti = np.load("/home/tushar/work/SVMB/final_run/seg_salt/seg_saltMods.npz")
	ti = ti['data']
	i = 0
	for test_num in test_nums:
		#tr_train, ti_train, tr_test, ti_test = trainTestSplit(tr, ti, test_num)
		np.savez_compressed("/home/tushar/work/SVMB/final_run/seg_salt/train_test/set{}_trainTraces.npz".format(i), data=tr[test_num:])
		np.savez_compressed("/home/tushar/work/SVMB/final_run/seg_salt/train_test/set{}_trainTiles.npz".format(i), data=ti[test_num:])
		np.savez_compressed("/home/tushar/work/SVMB/final_run/seg_salt/train_test/set{}_testTraces.npz".format(i), data=tr[:test_num])
		np.savez_compressed("/home/tushar/work/SVMB/final_run/seg_salt/train_test/set{}_testTiles.npz".format(i), data=ti[:test_num])

		i += 1

saltSEG_trainTestSplit(test_nums=[350, 500, 700, 1000])