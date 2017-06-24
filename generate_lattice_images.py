"""
GenerateLatticeImages
~~~~~~~~~~~~~~~~~~~~~

A small library of functions to produce simulated images of
point sources, trapped on a square grid. This is inspired by
quantum gas microscopy experiments (https://arxiv.org/abs/1510.04744),
in which fluorescence images are obtained of ultracold atoms trapped
in an optical lattice.

Graham Edge
November 15th, 2016
"""

#### Libraries
# Standard library
import gzip
import cPickle as pickle

# Third-party libraries
import numpy as np
import matplotlib.pyplot as plt



def generateImageStack(nTrain, nVal, nTest, name , nPix=13, AtomCounts = 100, AtomSTD = 1.32, shiftMag = 0.0, gridsize = 3, latticeAngle = 0.0):
	'''
	Generate multiple images of atoms sitting in a gridsize*gridsize square grid,
		imaged onto a sensor with nPix*nPix pixels. The images produced are represented
		by numpy arrays, and saved using cPickle as a .p file for further manipulation
		in Python.

	Inputs are the numbers of training, validation, and test images to output, 
		and a name for the saved pickle file.
	
	The data saved is a tuple containing three elements:
		- training_data, validation_data, test_data
	Each of these elements is a tuple containing two ndarrays with the length of nTrain,
		nVal, or nTest as applicable:
	 		- the first element of the tuple has as each element an ndarray 
	     		containing a simulated image pixeldata in 1D form 
	     		(so has length nPix^2)
	 		- the second element of the tuple has as each element an ndarray
	     		containing the lattice filling used to simulate the image in 1D form
	     		(so has length gridsize^2)

	This choice of data structure closely follows the MNIST data format used in 
		Michael Nielsen's book 'Neural Networks and Deep Learning". This makes it
		possible to adapt the sample code that accompanies the book to train a neural 
		net with these simulated images.

	The MNIST data would have as the classification a single number to label 
		the correct digit from 0-9
	Here the classification data is the specification of filled vs. empty lattice sites.
		For faster calculation of the net's accuracy, the validation and test data 
		could be converted from a lattice vector to a single digit identifier (again
		similar to the way Michael Nielsen handles the MNIST data), but this has not
		been implemented here.
	'''

	nImages = nTrain+nVal+nTest
	imageList = []
	fillingList = []
	
	#Fill a list with training images
	for n in xrange(nTrain):
		if n%100 == 0:
			print(str(n) + ' training images finished...')
		img, fillingGrid = generateImage(chipsize=nPix,AtomCounts = AtomCounts,AtomSTD = AtomSTD, 
				shiftMag = shiftMag, gridsize = gridsize, latticeAngle = latticeAngle)
		
		imageList.append(np.reshape(img, (nPix*nPix,)))
		fillingList.append(np.reshape(fillingGrid, (gridsize**2,)))

	valTestList = []
	valTestFillingList = []
	#Fill lists for validation data and test data (might do this differently to make accuracy simple to determine)
	for n in xrange(nVal+nTest):
		if n%100 == 0:
			print(str(n) + ' validation/test images finished...')
		img, fillingGrid = generateImage(chipsize=nPix,AtomCounts = AtomCounts,AtomSTD = AtomSTD, 
				shiftMag = shiftMag, gridsize = gridsize, latticeAngle = latticeAngle)
		
		valTestList.append(np.reshape(img, (nPix*nPix,)))
		valTestFillingList.append(np.reshape(fillingGrid, (gridsize**2,)))

	#Convert to numpy format
	imageStack = np.asarray(imageList)
	fillingStack = np.asarray(fillingList)
	valTestStack = np.asarray(valTestList)
	valTestFillingStack = np.asarray(valTestFillingList)

	#Separate the images to be used for training, validation, and testing
	trainingStack = ( imageStack, fillingStack)
	validationStack = ( valTestStack[0:nVal] , valTestFillingStack[0:nVal] )
	testStack =( valTestStack[nVal:] , valTestFillingStack[nVal:] )
		
	#Save the data as a pickled file
	with open(name, 'wb') as f:
		pickle.dump((trainingStack, validationStack, testStack), f)
	

def generateImage(chipsize=13, AtomCounts = 100, AtomSTD = 1.32, shiftMag = 0.0, 
		gridsize=3, filling = 0.5, latticeAngle = 0.0):
	'''
	Script to generate a simulated image, closely approximating the experimental conditions
		for pixelsize, lattice spacing, and PSF size.

	PSF is assumed to be a Gaussian.

	Parameters:
	chipsize 	-	number of pixels in each dimension of the square image
	AtomCounts 	-	number of photons that are collected from each atom on average
	AtomSTD		-	standard deviation of the Gaussian PSF, in units of pixels
	shiftMag 	-	magnitude (in units of lattice sites) with which to shift the lattice
						around randomly when generating images, to simulate experimental uncertainty
						in the lattice grid placement
	gridsize 	-	the number of lattice sites along each dimension of the square lattice grid
	filling 	-	a fraction of sites to fill with atoms (to assess performance in the 
						limits of sparse, half-filled, and fully-filled lattices)
	
	In the real experiment (https://arxiv.org/abs/1510.04744) the length scales are:
		- pixelsize 		=	195nm
		- lattice spacing 	=	527nm	=	2.7px
		- PSF FWHM 			=	607nm
		- PSF Standard Dev 	=	258nm 	=	1.32px
	'''

	#Some hard-coded parameters for testing
	oversample=10		#the factor by which we oversample the image before drawing from the Gaussian PSF
	shifting = True
	fullFilling = True
	latticeSpacing = 1*2.7
				#Later we will generate images with lattice grid rotated
								# 	with respect to pixel coordinates

	#Generate blank image arrays
	img = np.zeros((chipsize,chipsize))
	imgOversample = np.zeros((chipsize*oversample, chipsize*oversample))
	anchor = np.asarray([np.floor(chipsize/2),np.floor(chipsize/2)])
	

	#Lattice Vectors
	dx = [np.cos(latticeAngle)*latticeSpacing, \
				np.sin(latticeAngle)*latticeSpacing]
	dy = [-np.sin(latticeAngle)*latticeSpacing, \
				np.cos(latticeAngle)*latticeSpacing]
	
	#We want to classify the inner gridsizexgridsize sites of a larger distribution, so we will
	#	expand the number of sites that may hold an atom (by an extra 2 sites on each side)
	latticesize = gridsize+4
	

	#Initialize lattice filling
	fillingGrid = np.zeros((latticesize,latticesize))


	#Check whether we are arbitrarily filling the lattice (fullFilling = 1)
	#	or filling to an arbirary filling fraction
	if fullFilling:
		#Atoms possible anywhere in the grid
		fillingGrid=np.round(np.random.random(size=(fillingGrid.shape))).astype(int)
	elif filling == 1.0:
		#Atoms are everywhere in the grid
		fillingGrid=np.ones((latticesize,latticesize))
	else:
		#Full up to a fixed number of atoms (currently slow, but not usually necessary)
		nAtoms = np.round(filling * (gridsize**2))
		n=0
		while n < nAtoms:
			i = np.round((latticesize-1)*np.random.random()).astype(int)
			j = np.round((latticesize-1)*np.random.random()).astype(int)
			if fillingGrid[i,j]==0:
				fillingGrid[i,j] = 1
				n = n+1

	#Fill in the outer corners to anchor
	#fillingGrid[0,0] = 1

	#Now crop out the central gridsizexgridsize sites to serve as a 
	#	label for the image
	fillingLabel = fillingGrid[2:-2, 2:-2]
		#python indexing [2:-2] is every element except the first two and last two

	#Make the lattice grid 1D
	filling = fillingGrid.flatten()
	

	#Create meshgrid variables for lattice indices
	II, JJ = np.meshgrid(np.arange(fillingGrid.shape[0]), \
							np.arange(fillingGrid.shape[1]))


	#Reference to grid centre
	II = II - np.floor(latticesize/2)
	JJ = JJ - np.floor(latticesize/2)


	#Add random offset from the grid centre
	if shifting:
		#Uniform distribution of shifts
		#II = II - shiftMag*(2*np.random.random() - 1)
		#JJ = JJ - shiftMag*(2*np.random.random() - 1)

		#Fixed shift size for all images
		theta = 2*np.pi*np.random.random()	#Choose a random angle
		r = shiftMag	#Fixed shift size in pixels
		anchor = anchor + np.asarray([r*np.cos(theta), r*np.sin(theta)])

		#Gaussian Distributed Anchor Position (like what can be determined experimentally)
		#	we measure sigma = 60nm = 0.31px  
		#sigmaAnchor = shiftMag 	#0.31 for experimental
		#anchor = anchor + np.random.normal(loc=0, scale=sigmaAnchor, size=2)

	#List of filled sites
	Ifilled = II.ravel()[np.flatnonzero(fillingGrid)]
	Jfilled = JJ.ravel()[np.flatnonzero(fillingGrid)]
	

	#Create a list of pixel coordinates for filled lattice sites, referenced to Anchor
	AtomPos = [ (Ifilled*dx[0] + Jfilled*dy[0]) + anchor[0] , (Ifilled*dx[1] + Jfilled*dy[1]) + anchor[1]]

	#Fill Lattice
	img = fillLattice(img, imgOversample, oversample, \
					AtomPos, AtomSTD, AtomCounts)
	

	#Add Noise
	img = addNoise(img).astype(int)


	# #Rescale to have max roughly equal to one (in absence of noise)
	peakCounts = AtomCounts / (2*np.power(AtomSTD,2)*np.pi)	#max of a Gaussian function
	img = img / peakCounts


	return img, fillingLabel



def fillLattice(img, imgOversample, oversample, \
				AtomPos, AtomSTD, AtomCounts):
	"""
	Generates fluorescence counts in the pixels of the image 'img'
		based on the given lattice filling 'AtomPos' - a tuple specifying
		the x and y positions of each atom as lists
	The shape of each blurred atom image is given by a Gaussian point-spread 
		function, with standard deviation 'AtomSTD', and integrated signal
		'AtomCounts'

	'oversample' specifies the number of subpixels to break each pixel into before
		sampling from the Gaussian function
	'imgOversample' is a black array in which to store the oversampled image	
	"""
	XX, YY = np.meshgrid(np.arange(imgOversample.shape[0]), \
						np.arange(imgOversample.shape[1]))
						
	gaussPeak = AtomCounts / (2*np.power(AtomSTD,2)*np.pi)
	
	for Xc, Yc in zip(AtomPos[0]*oversample, AtomPos[1]*oversample):
		#This adds photons to each pixel in the image, based on an atom at position Xc, Yc
		#	(this still makes sense even if Xc and Yc are not inside the image!)
		imgOversample = imgOversample + gaussPeak*gaussian(XX,YY,Xc,Yc,AtomSTD*oversample)
	
	img = binImage(imgOversample, oversample, oversample)
	
	return img
	


def addNoise(img):
	"""
	Takes an image 'img' and returns the same image containing all 
		expected noise sources found in a quantum gas microscope
		experiment using an EMCCD camera:
			- shot noise
			- EMCCD gain enhancement of shot noise
			- dark counts
			- clock-induced charges

	For each pixel in the image, we have an incident number of photons.
		We calculate the detected number by drawing from a Poisson
		distribution, which results in tha addition of shot noise.

	We simulate the EMCCD noise by drawing from a Poisson with mean
		lambda/2 rather than lambda, and then scaling up by 2. This is not 
		mathematically correct, but is a good approximation for large photon numbers.

	Dark counts on the camera sensor are added by drawing from a 
		poisson distribution with mean 1/8 photon

	Clock-induced charges can be included by drawing from a Gaussian
		distribution with small mean value. This is still to be implemented.	
	
	This function should be improved by drawing from the experimentally
		measured noise distribution from bacground images.
	"""

	#Shot noise for the detected photons
	img = 2*np.random.poisson(img/2)
	
	#Add dark counts
	img = img + np.random.poisson(np.ones(img.shape)/8)
	
	#Add Gaussian noise for clock errors?
	#
	
	return img


def xy2IJ(x,y,dx,dy):
	"""
	Takes pixel coordinates x and y, and produces corresponding lattice 
		site indices I and J using the lattice grid defined
		by the tuples dx and dy
	"""
	I = ((x*dy[1]-y*dy[0])/(dx[0]*dy[1]-dx[1]*dy[0]))
	J = ((X*dx[1]-Y*dx[0])/(dy[0]*dx[1]-dy[1]*dx[0]))
	return I, J
	

def IJ2xy(I,J,dx,dy):
	"""
	Takes lattice indices I and J, and produces corresponding pixel 
		coordinates x and y using the lattice grid defined
		by tuples dx and dy.
	"""
	x = (I*dx[0] + J*dy[0])
	y = (I*dx[1] + J*dy[1])
	return x, y
	

def gaussian(x, y, xc, yc, s):
	'''Calculate the magnitude, at coordinates specified by 'x','y',
		 of a 2D gaussian function centred at 'xc','yc' with 
		 amplitude 1, and symmetric standard deviation 's'
	''' 
	return np.exp(-(np.power(x - xc, 2.) + np.power(y - yc, 2.))/ (2 * np.power(s, 2.)))
	
	
def binImage(img, binx, biny):
	'''Bin the pixels of the image 'img' into superpixels,
		with 'binx' and 'biny' specifying the bin sizes
	'''
	binnedOne = np.zeros((img.shape[0]/binx, img.shape[1]))
	binned = np.zeros((img.shape[0]/binx, img.shape[1]/biny))
	for i in xrange(0, img.shape[0],binx):
		binnedOne[i/binx,:] = np.mean(img[i:i+binx,:],axis=0)
	for j in xrange(0, img.shape[1],biny):
		binned[:,j/biny] = np.mean(binnedOne[:,j:j+biny], axis=1)
	return binned


def lattice2Bin(lattice):
	'''Takes a lattice occupation vector (zeros and ones) and returns a unique
	number to identify this filling'''
	labels = np.asarray([2**n for n in np.arange(len(lattice))])	#binary values [1 2 4 8 16 32 ...]
	return np.dot(lattice, labels)


def bin2Lattice(bin, nSites):
	'''Takes the unique number that identifies the lattice and translates back
	into the occupation of each site

	Currently hard-coded for 3x3 site lattice grid!!
	'''
	num_bits = 8
	bin_format = '08b' 
	sitelist = []
	for char in format(bin, bin_format):
		sitelist.append(int(char))
	return np.asarray(sitelist)
	

def plotImages(img, filling):
	'''
	Just a simple plot of the generated image, alongside the corresponding lattice filling
		to be used for debugging purposes
	'''
	latticeAngle = 30*3.14/180
	latticeSpacing = 2.7

	imgSize = img.shape[0]
	gridsize = filling.shape[0]

	dx = [np.cos(latticeAngle)*latticeSpacing, \
				np.sin(latticeAngle)*latticeSpacing]
	dy = [-np.sin(latticeAngle)*latticeSpacing, \
				np.cos(latticeAngle)*latticeSpacing]

	ISites = np.asarray([-1, 0, 1, -1, 0, 1, -1, 0, 1])
	JSites = np.asarray([-1, -1, -1, 0, 0, 0, 1, 1, 1])

	anchor = np.asarray([np.floor(imgSize/2), np.floor(imgSize/2)])
	xSites = (ISites*dx[0] + JSites*dy[0]) + anchor[0]
	ySites = (ISites*dx[1] + JSites*dy[1]) + anchor[1]

	plt.subplot(121)
	plt.imshow(img,interpolation='none')
	plt.plot(xSites,ySites, 'ow', fillstyle='none', markersize='10', markeredgewidth='1' )			#Highlight the centres of lattice sites
	plt.colorbar()
	plt.xlabel('x (px)')
	plt.ylabel('y (px)')
	plt.xlim(-0.5, imgSize-0.5)
	plt.ylim(-0.5, imgSize-0.5)
	plt.title('Simulated Image')
	
	plt.subplot(122)
	plt.imshow(filling, interpolation='none', clim = (0,1))
	plt.title('True Atom Filling')
	plt.xticks(np.arange(0,filling.shape[0],1))
	plt.yticks(np.arange(0,filling.shape[1],1))
	plt.xlabel('x (sites)')
	plt.ylabel('y (sites)')
	plt.xlim(-0.5, gridsize-0.5)
	plt.ylim(-0.5, gridsize-0.5)
	plt.colorbar()

	plt.tight_layout()

	plt.show()