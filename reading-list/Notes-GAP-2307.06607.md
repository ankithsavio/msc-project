Comprehensive Notes for [[2307.06607v2.pdf]]

- Fluorescence microscopy 
	- Excitation light : light of specific wavelength
	- Fluorophores : molecules that absorb the excitation light and emit light of higher wavelength (different color)
	- Fluorescence : emitted light from the fluorophores
	- Uses : can target-visualize-specific parts of a sample
	- Disadvantages : 
		- Fluorophores are delicate, can undergo photobleaching and lose ability to fluoresce when exposed to too much light
		- Phototoxicity : damage due to high intensity light to cells causing death

- Noise in Fluorescence microscopy due to limiting the light to avoid damaging the sample
- Low light condition lead to physically inevitable noise - Poisson shot noise
	- Shot noise : fluctuation in the number of photons detected due to their randomness or independent nature.
	- Photons follow a Poisson distribution

- Traditional method : 
	- Supervised Learning : Noise -> Clean
		- Autoencoders
	- Self-Supervised Learning : 
		- Noise2Noise : minimizing MSE for pair of Noisy images, the model learns to denoise the image
		- Noise2Void : using a single noise image to denoise image, assume pixels predictable from surrounding pixels but not the noise (noise are pixel-independent). drop the center pixel from the receptive field and train the model to predict it. the model learns to denoise in the similar fashion
- Problem : Not possible to recover the true underlying clean image.
- Proposal : Image is an generative accumulation of photons and are perceptible to shot noise
	- Train CNN to predict a distribution for the next possible photon location
	- Predicting next photon $\approx$ denoising
	- 

