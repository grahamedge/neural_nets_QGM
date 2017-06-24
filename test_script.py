"""NetworkTrainScript.py
~~~~~~~~~~~~~~

A simple script to run the Theano neural network code 
	'network3Lattice.py' on a batch of simulated fluorescence images

The folder containing the images is hard coded, as well as the name
	of the pickled list of Image Dataset names


Graham Edge
March 1st, 2017
"""


#### Libraries
# Standard library
import cPickle
import re

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt

# Nielsen book libraries
import network
from network import FullyConnectedLayer, ConvPoolLayer, SoftmaxLayer
from network import Network


#Script
#-----------------

#Flags for plotting or saving the network training results
plotting = False 	#plotting works best for 3x3 clusters at the moment
saving = True 		#saves a numpy array of the training results to a pickled file

#Hard coded foldername and filename for the stackNames pickled file
foldername = '/home/graham/Python/Theano/LatticeFilling/Data/3x3Grid_AnyFilling_Surrounded/'
namefile = foldername + '3x3Grid_AnyFilling_Surrounded_names.p'

#Parameters of the dataset (in the future should be read 
# 			from the data rather than hard-coded)
imagesize = 13		#number of pixels
gridsize = 3 		#number of lattice sites


#Parameters to use for training the neural net
nEpochs = 100 			#number of epochs to train for
mini_batch_size = 10 	#size of each subset of data used for training
eta = 1				#parameter controlling the step size for stochastic gradient descent
lmb = 0.0				#regularization parameter used to avoid overfitting

#Parameters specific to a Single Layer Convolutional Neural Net (otherwise ignored)
nFilters = 15
filtersize = 15
pool = 1
if (imagesize+1-filtersize) % pool != 0:
	print('Cannot pool the filtered images due to odd dimensions. Change filter size!')
else:
	pooledImageSize = (imagesize+1-filtersize)/pool


#Generate a string to save the training results as a pickled file
saveName = 'FC_Network_Results' + str(eta) + '_Lambda' + str(lmb) + '_Epochs' + str(nEpochs) + 'BatchSize_' + str(mini_batch_size) + '.p'
saveloc = foldername + saveName

#Open the list of dataset names
with open(namefile, 'rb') as f:
	file_list = cPickle.load(f)

#Tran only some of the files?
file_list = file_list
print('Files to be trained:')
print(file_list)

#Create an empty array for the training results:
# 	5 columns for: nPhotons, accuracy, std, shift_size, accuracy_per_atom
results = np.zeros((len(file_list), 5))

#Find the string locations of the parameters to be read
read_params = ['counts', 'width', 'shift']
param_index = []
filename = file_list[0] + '.p'
params = filename.split('_')
for idx, rp in enumerate(read_params):
	indices = [n for n, s in enumerate(params) if read_params[idx] in s]
	if len(indices)>1:
		print('Too many matching indices found for read params. First will be assumed!')
	param_index.append(indices[0])
	

#Loop over all datasets in the list of datasets
results = {}	#initialise a dictionary to hold results
for n, file in enumerate(file_list):

	#Set up filename and load data
	#-------------------------------

	#Construct path to the file
	filename = file
	path = foldername + filename

	#Split the file name in order to extract some parameters
	#	(this is a very poor way of doing this, should just save
	#	the parameters alongsize the dataset... not fixed through because
	#	I want some practice using RegEx)
	params = filename.split('_')
	try:
		nPhotons = int(re.findall(r'\d+', params[param_index[0]])[0])
		std = float(re.findall(r'\d\.\d+', params[param_index[1]])[0])
		shift = float(re.findall(r'\d+', params[param_index[2]])[0])
		print('Found parameters %d, %f, %d' % (nPhotons, std, shift))
	except:
		print('Error extracting data from filename:')
		print(params)

	#Output to terminal to follow along with the training
	print('Training file ' + str(n) + ': ' + str(nPhotons) + ' photons, ' + 
		 str(std) + 'px standard deviation, with ' + str(shift) + 'pixel shifting.\n')

	#Load data
	training_data, validation_data, test_data = network3Lattice.load_data_shared(path)
	
	#The data for the network is stored in Theano shared variables, but we want to be
	#	able to compare network output to the real files, so we create a backup version
	training_backup, validation_backup, test_backup = network3Lattice.load_data(path)


	#Construct an instance of the network class, specifying the structure of
	#	convolutional layers, fully connected layers, softmax layers, etc...
	#	(softmax layers not really appropriate for this training task, so we use 
	# 	a fully-connected output layer with as many neurons as lattice sites)
	#------------------------------------------------------------------------

# #CNN with 3 Fully Connected Layers
# 	net = Network([
# 		ConvPoolLayer(image_shape=(mini_batch_size, 1, imagesize, imagesize), 
# 			filter_shape = (nFilters, 1, filtersize,filtersize), poolsize=(pool,pool)),
# 		FullyConnectedLayer(n_in = nFilters*pooledImageSize**2, n_out = 50),
# 		FullyConnectedLayer(n_in = 50, n_out = 18),
# 		FullyConnectedLayer(n_in = 18, n_out = 9)], mini_batch_size)

#CNN with 2 Fully Connected Layers
	# net = Network([
	# 	ConvPoolLayer(image_shape=(mini_batch_size, 1, imagesize, imagesize), 
	# 		filter_shape = (nFilters, 1, filtersize,filtersize), poolsize=(pool,pool)),
	# 	FullyConnectedLayer(n_in = nFilters*pooledImageSize**2, n_out = 300),
	# 	FullyConnectedLayer(n_in = 300, n_out = 9)], mini_batch_size)

# CNN with 1 FC Layer
	# net = Network([
	# 	ConvPoolLayer(image_shape=(mini_batch_size, 1, imagesize, imagesize), 
	# 		filter_shape = (nFilters, 1, filtersize,filtersize), poolsize=(pool,pool)),
	# 	FullyConnectedLayer(n_in = nFilters*pooledImageSize**2, n_out = gridsize**2)],mini_batch_size)

#2 Layer FC Net
	# net = Network([
	# 	FullyConnectedLayer(n_in = imagesize**2, n_out = 30),FullyConnectedLayer(n_in = 30, n_out = 9) ], mini_batch_size)

#1 Layer FC Net
	net = Network([
		FullyConnectedLayer(n_in = imagesize**2, n_out = gridsize**2) ], mini_batch_size)



	#Attempt Stochastic Gradient Descent
	#-----------------------------------
	weights, biases = net.SGD(training_data, nEpochs, mini_batch_size, eta, validation_data, test_data, lmbda = lmb)

	for N,w in enumerate(weights):
		print('Weights for layer %d have shape:' %N)
		print(w.shape)
	for N, b in enumerate(biases):
		print('Biases for layer %d have shape:' %N)
		print(b.shape)

	#Save data
	#---------

	#Save results as a dictionary!
	print(n)
	print(results.keys())
	results[n] = {'Photons':nPhotons, 'Std':std, 'Grid Shift':shift,\
				 'Perfect Images':net.finalAccuracy, 'Atom Accuracy':net.perAtomAccuracy,\
				 'Weights':weights, 'Biases':biases}

	#Plotting
	#---------
	if plotting:
		#Choose a group of the test images, and examine the network classification
		#	versus the input image. Useful for debugging a network that doesn't train

		#Hard-coded to always look at the first 10 images currently
		i_start = 0
		i_stop = i_start+10

		#Pull the inputs, activations, predictions from the network
		inputs = net.test_mb_inputs(0).astype('f2')
		inputs = inputs[i_start:i_stop]
		activations = net.test_mb_outputs(0).astype('f2')
		activations = activations[i_start:i_stop]
		predictions = np.round(activations).astype(int)
		predictions = predictions[i_start:i_stop]

		#Pull the correct lattice fillings from the backed-up version 
		#	of the test dataset
		answers = 	test_backup[1][i_start:i_stop]
		atomAccuracy = net.test_atom_accuracy(0).astype('f2')


		#Plot results with matplotlib
		plt.figure(figsize=(20,10), dpi=80, facecolor='w', edgecolor='k')

		latticeAngle = 30*3.14/180
		latticeSpacing = 2.7

		dx = [np.cos(latticeAngle)*latticeSpacing, \
					np.sin(latticeAngle)*latticeSpacing]
		dy = [-np.sin(latticeAngle)*latticeSpacing, \
					np.cos(latticeAngle)*latticeSpacing]

		ISites = np.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1])
		JSites = np.asarray([-1, -1, -1, 0, 0, 0, 1, 1, 1])

		anchor = np.asarray([np.floor(imagesize/2), np.floor(imagesize/2)])
		xSites = (ISites*dx[0] + JSites*dy[0]) + anchor[0]
		ySites = (ISites*dx[1] + JSites*dy[1]) + anchor[1]

		for plotnumber in range(i_start,i_stop):
			#Use subplots to look at 10 images simultaneously

			#Check whether the prediction for this image was correct
			correct = (predictions[plotnumber] == answers[plotnumber])
			if correct.all():
				#correct.all() is only true if EVERY lattice site is correctly labelled
				col = 'green'
			else:
				col = 'red'

			xFilledSites = xSites[answers[plotnumber]==1]
			yFilledSites = ySites[answers[plotnumber]==1]

			plt.subplot(2,5,plotnumber+1)
			plt.imshow(test_backup[0][plotnumber].reshape((imagesize,imagesize)),interpolation='none', clim=(0.0, 1.2))
			plt.plot(xSites,ySites, 'ok',fillstyle='none',  markersize='10', markeredgewidth='1.4' )					#Highlight the centres of lattice sites
			plt.plot(xFilledSites, yFilledSites, 'ow', markersize='6')		#Intensely highlight the centres of filled sites
			plt.xlabel('x (px)')
			plt.ylabel('y (px)')
			plt.xlim(-0.5, imagesize-0.5)
			plt.ylim(imagesize-0.5, -0.5)

			#Generate a title for each subplot, manually hacking in
			#	a visualization of the filling matrix... this might
			#	be better to move into an inset plot for more flexibility
			plt.title('Predicted Filling\n' + 
				str(predictions[plotnumber][:3]) + '\n' + 
				str(predictions[plotnumber][3:6]) + '\n' + 
				str(predictions[plotnumber][6:9]), color = col)

		#plt.suptitle('Convolutional Net Results for ' + str(nPhotons) + ' Photons and $\sigma = $' + str(std) + '.')
		plt.tight_layout()
		

		plt.show()

if saving:
	with open(saveloc, "wb") as f:
		cPickle.dump(results, f)