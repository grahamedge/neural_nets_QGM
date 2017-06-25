# Neural Networks for Quantum Gas Microscopy

This repository contains some python code that I used to perform some image analysis in my PhD research at the University of Toronto. I found that neural networks can be trained to recognize the presence of very dim sources of light in an image, even when the sources of light are so close together as to be considered 'unresolved'. The performance was surprisingly good, and seemed to match or surpass the statistical fitting procedures (based on Maximum Likelihood Estimation) that we had originally developed to perform image analysis on our experimental data.

## Quantum Gas Microscopy?

My PhD was in the field of experimental quantum physics, and my research was focussed on the contruction of a machine called a quantum gas microscope. The design goal of this machine is to produce a sample of a few hundred atoms at incredibly low temperatures, trap them in a square grid called an "optical lattice", and then image the positions of the atoms one by one with a powerful microscope.

Making an experiment like this is only really worth doing if the square grid that holds the atoms is very finely spaced - atoms trapped in the grid should be separated by distances not much greater than the wavelength of visible light (400-700nm). This small separation means that atoms can quantum-mechanically tunnel from one point in the lattice grid to another (this is the reason for the term "quantum gas" in "quantum gas microscopy").  The downside of the small separation is that when we try to produce an image of the atoms in the lattice grid, we need an imcredibly powerful microscope to be able to answer the question "what atoms are sitting in what sites of the grid". Even when we use a great microscope objective, the atoms in our experiment are closer together than the "minimum resolvable distance". To analyse such an "unresolved" image, we need to perform an image analysis which is aware of all the additional prior information that we have about the experiment.

## Maximum Likelihood Estimation?

To analyse the images that we produce with the quantum gas microscope experiment, we built a maximum likelihood estimation model. We know that if atoms are present in an image, they must all be pinned in the sites of the same square grid. This prior information is incredibly important. It would be incredibly hard to answer the question "what are the positions of the atoms in this image" if the atoms could appear anywhere. Since the atoms can only appear on specific sites of the square grid, we instead are answering the question "which sites in the grid seem to have an atom in them, and which seem to be empty". 

To answer this question, we use our extensive knowledge of the grid and the imaging system to simulate what an image would look like for some specific, pre-defined filling of the lattice.  We can then compare the image that we observe with the simulated image, to decide whether the lattice filling that we used to generate the image is a good guess for the actual lattice filling that occurred in the experiment. Since the image that we observe isn't produced deterministically, but instead is subject to many different kinds of noise, when we want to determine whether a specific lattice distribution is a good match to the observed image we need to ask whether the observed and expected images are similar to within the expected noise distributions. This is where maximum likelihood estimation comes into play. 

Since we have a lot of prior knowledge about the nature of the noise distributions, we can estimate the likelihood that an individual observation (the experimental image) is consistent with being drawn from a specific probability distribution (the specific distribution determined by a certain set of filled and empty sites in the lattice grid). Once we can determine this likelihood of an observed image being an example of a specific distribution, we can try to find the specific distribution that would give the maximum likelihood of producing the image that we observed.  This is the essence of the aximum likelihood model.

## Neural Networks

Running the maximum likelihood estimation model on each image produced by the experiment took a long time, and it was still very difficult to tweak the thresholds in the model to achieve good accuracy in the image analysis. When I later heard about how successful convolutional neural networks could be in image analysis tasks, I got really curious to see whether such a network could do a good job of analysing the poorly resolved images of atoms that were produced in my experiment.

## Training Data

To train a neural network, one needs to have a lot of example data with known classifications. The images produced by my experiment are not at all suitable for training the network, because there are not enough of them and because we don't know the true distribution of atoms that gives rise to a given image. Luckily, we know so much about the imaging apparatus that we use that we can generate simulated images for any hypothetical filling of the lattice grid. These simulated images are not a "cheat" version of of the real experimental data... we built into the simulation all of the known noise distributions: finite image resolution, shot noise due to the small number of photons that we collect, statistical noise due to the electron multiplication step in our camera sensor, clock induced charge noise introduced in the camera readout process, and the rare detection of background light on our camera sensor.  Including all of these sources produces an incredibly close approximation to the true experimental data, with the bonus that I can generate millions of such images per day (rather than tens of real images for an average day in the laboratory).

For a small 3x3 lattice grid, I generated tens of thousands of images that covered all of the possible 2^9 ways the lattice could be filled. For each simulated image I knew the true filling that was used to produce it, and I fed these images and truth values into various neural network architectures to see whether they could learn to recognize the underlying structure in the images despite the large amounts of statistical noise. I wrote all of the neural network code in python, and based everything off of the examples provided in Michael Nielsen's book [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). The examples in the book are for handwritten digit recognition on the MNIST dataset, and use a softmax layer to output the probability of each of the ten possible digits. In my case I had 2^9 possibile outcomes, and so I had two options: make the output layer have 2^9 neurons ("one-hot encoding" each of the different possibilities) or change the cost function to assess how many of the 9 lattice sites in the 3x3 grid had been correctly identified.  I opted for the latter since the scaling is much better for increases in the lattice grid size, and ended up implementing a cross-entropy cost function that is averaged over all of the sites in the lattice grid.

## Training the Network

The code I used was written to allow convolutional network architectures. However, I found that I could get great results by using only shallow fully connected networks. This was actually a big relief because I was training all of the networks on my laptop CPU, and wouldn't have been able to get very far if deep convolutional networks were required.  In the end quite good results could be obtained by using only a single layer fully connected network, with weights connecting each pixel in the input image to each of the 9 possible output neurons that represent the 3x3 lattice grid. The single-layer structure of the network means that no nonlinear terms which combine different pixels can occur. What the single-layer network is really learning is akin to a complex image filter that maps an input image to a desired output. Once the network has been trained, applying it to new images is incredibly fast and straightforward:
1. multiply the pixel values by a matrix of weights
2. add a vector of biases to the 9 outputs
3. apply the sigmoid function
4. round the resulting 9 output values to get a prediction 1 (site has an atom) or 0 (site is empty)

The accuracy with which the trained network finds the true distribution is as good or better than the maximum likelihood model that was developed previously. Furthermore, the actual process of getting a prediction from an input image is much faster than applying the maximum likelihood fitting algorithm!
