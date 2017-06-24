	"""PlotResults.py
~~~~~~~~~~~~~~

A simple script to plot the results of Theano neural network code 
	'network3Lattice.py' after training on several sets of images


Graham Edge
March 15th, 2017
"""

#### Libraries
# Standard library
import cPickle

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import network
import deepdish	as dd	#For saving HDF5 data


#define a sigmoid fuction so that I can calculate the 
#	network output based on a set of weights and biases
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Hard coded filenames
#-------------------

#MLE Results Location
MLEfolder = '/home/graham/Python/Theano/LatticeFilling/MLE Data/'
MLEfile = MLEfolder + 'FixedGrid_MLE_Errors.csv'

Allfoldername = '/home/graham/Python/Theano/LatticeFilling/Data/3x3Grid_AnyFilling_Surrounded_GaussianShifts/'
Allresults_file = Allfoldername + 'FC_Network_Results1_Lambda0.0_Epochs100BatchSize_10.p'
FCnEpochs = 100
FClmb=0.0
FCeta = 0.1

with open(Allresults_file, 'rb') as f:
	ShiftResults = cPickle.load(f)

Allfoldername = '/home/graham/Python/Theano/LatticeFilling/Data/3x3Grid_AnyFilling_Surrounded/'
Allresults_file = Allfoldername + 'FC_Network_Results1_Lambda0.0_Epochs100BatchSize_10.p'

with open(Allresults_file, 'rb') as f:
	FixedResults = cPickle.load(f)

#Load data
#---------
dfMLE = pd.read_csv(MLEfile)
dfMLE.columns = ['Photons', 'Accuracy']


#Results are stored as a dictionary... so unpack them into an array for plotting
Fixed = np.zeros((len(FixedResults.keys()), 3))
Shift = np.zeros((len(ShiftResults.keys()), 3))
print(FixedResults.keys())

for i, key in enumerate(FixedResults.keys()):
	Fixed[i,0] = FixedResults[key]['Photons']
	Fixed[i,1] = FixedResults[key]['Perfect Images']
	Fixed[i,2] = FixedResults[key]['Atom Accuracy']
	FixedWeights = FixedResults[key]['Weights']
	FixedBiases = FixedResults[key]['Biases']

	print(FixedWeights[0].shape)

for i, key in enumerate(ShiftResults.keys()):
	Shift[i,0] = ShiftResults[key]['Photons']
	Shift[i,1] = ShiftResults[key]['Perfect Images']
	Shift[i,2] = ShiftResults[key]['Atom Accuracy']
	ShiftWeights = ShiftResults[key]['Weights']
	ShiftBiases = ShiftResults[key]['Biases']

#Output as .csv these weights and biases
saveNetwork = True
if saveNetwork:
	np.savetxt("/home/graham/Python/Theano/LatticeFilling/WeightMatrix.csv", FixedWeights[0], delimiter=",")
	np.savetxt("/home/graham/Python/Theano/LatticeFilling/BiasVector.csv", FixedBiases[0], delimiter=",")



	network = {'weights':ShiftWeights, 'biases':ShiftBiases}
	dd.io.save('/home/graham/Python/Theano/LatticeFilling/Network.h5', network)

#Save some example images in the HDF5 format too, to play around with
if saveNetwork:

	foldername = '/home/graham/Python/Theano/LatticeFilling/Data/3x3Grid_AnyFilling_Surrounded_GaussianShifts/'
	namefile = foldername + '3x3Grid_AnyFilling_Surrounded_GaussianShifts_Sparse_250counts_1.32width_0.31shift_13px_30deg.p'
	trainImg, validImg, test_images = network3Lattice.load_data(namefile)

	ImageDict = {'Training_Images': trainImg[0], 'Training_Answers': trainImg[1], \
					'Validation_Images': validImg[0], 'Validation_Answers': validImg[1], \
					'Test_Images': test_images[0], 'Test_Answers': test_images[1]}

	image_zero = test_images[0][0]
	image_zero_square = image_zero.reshape((13,13))
	print(np.shape(image_zero_square))
	answer_zero = test_images[1][0]
	np.savetxt("/home/graham/Python/Theano/LatticeFilling/TestImage.csv", image_zero, delimiter=",")
	
	np.savetxt("/home/graham/Python/Theano/LatticeFilling/TestImageSquare.csv", image_zero_square, delimiter=",")

	np.savetxt("/home/graham/Python/Theano/LatticeFilling/TestAnswer.csv", answer_zero, delimiter=",")

	dd.io.save('/home/graham/Python/Theano/LatticeFilling/ShiftedTestImages.h5', ImageDict)

#Import some specific images to feed into the network
foldername = '/home/graham/Python/Theano/LatticeFilling/Data/3x3Grid_AnyFilling_Surrounded_ConstantShifts/'
namefile = foldername + '3x3Grid_AnyFilling_Surrounded_ConstantShifts_names.p'
with open(namefile, 'rb') as f:	
	file_list = cPickle.load(f)
ConstantShiftResults = np.zeros((len(file_list), 4))	
for n, file in enumerate(file_list):	
	path = foldername + file
	params = file.split('_')
	photons = int(re.findall(r'\d+', params[5])[0])
	shift = float(re.findall(r'\d.\d+', params[7])[0])
	print('Loading Images for %f shift' % shift)
	trainImg, validImg, test_images = network3Lattice.load_data(path)

#	#Cut the networks loose on these new images
	AtomAccuracy = np.zeros((len(test_images[0]),))
	PerfectImages = np.zeros((len(test_images[0]),))
	for N,image in enumerate(test_images[0]):
		output = image.dot(ShiftWeights[0]) + ShiftBiases[0]
		guess = np.round(sigmoid(output))

		AtomAccuracy[N] = np.mean(guess==test_images[1][N])
		PerfectImages[N] = np.all(guess==test_images[1][N])


	ConstantShiftResults[n,0] = photons
	ConstantShiftResults[n,1] = shift
	ConstantShiftResults[n,2] = np.mean(AtomAccuracy)
	ConstantShiftResults[n,3] = np.mean(PerfectImages)

print(ConstantShiftResults)


#Plot data
#---------

plt.figure(figsize=(12,8), dpi=80, facecolor='w', edgecolor='k')

nPhotons = 500
scale = 60.0/0.31

plt.plot(scale*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,1],\
			 100*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,2],
	'ob', markersize=10, linestyle='dashed', label='500 Photons')
nPhotons = 250
plt.plot(scale*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,1],\
			 100*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,2],
	'om', markersize=10, linestyle='dashed', label='250 Photons')
nPhotons = 150
plt.plot(scale*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,1],\
			 100*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,2],
	'or', markersize=10, linestyle='dashed', label='150 Photons')
nPhotons = 100
plt.plot(scale*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,1],\
			 100*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,2],
	'og', markersize=10, linestyle='dashed', label='100 Photons')
nPhotons = 50
plt.plot(scale*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,1],\
			 100*ConstantShiftResults[ConstantShiftResults[:,0]==nPhotons,2],
	'ok', markersize=10, linestyle='dashed', label='50 Photons')


plt.legend(loc='lower right', fontsize = 20)

plt.xlabel('Lattice Grid Error (nm)', fontsize = 20)
plt.ylabel(r'Per Atom Accuracy (%)', fontsize = 20)

# plt.xlim(((-5,505)))
plt.ylim((-1,104))
plt.xlim((-5, 235))
plt.grid(b=True, which='both', color='0.65',linestyle='-')

plt.title('Simple NN with Uncertainty in Lattice Grid\n[$\sigma=1.32\,$px]',
	fontsize = 24)
plt.show()