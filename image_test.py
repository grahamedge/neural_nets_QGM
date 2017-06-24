'''Small file to produce a simulated lattice image and plot it'''

import generate_lattice_images as imgen

I, F = imgen.generateImage(chipsize = 13, gridsize=3, AtomSTD = 0.5 , latticeAngle = 30*3.14/180, shiftMag = 1.0)
imgen.plotImages(I,F)