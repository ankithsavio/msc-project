Comprehensive Notes for 2307.06607v2.pdf

### Overview 
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

- Image : 
	- Formation :
		- Digital sensors : Have multiple detector elements 
		- Simplified model : No. of pixel = No. of detector elements
		- Ideal case : pixel value = No. of photons
	- Shot Noise :
		- No. of photons at each detector element or the pixel location follows a Poisson distribution
		- Photons are independent, Therefore we can use product of independent distributions to describe the whole noisy image

- Denoising
	- MMSE : Minimum MSE leads to expectation over all possible clean images, compromising on the quality of the images.


### Explanation of formulas
- Pixels follow Poisson distribution
	- $p(x_i \mid s_i) = \frac{s_i^{x_i} \exp(-s_i)}{x_i!}$
	- where $x_i$ is the noisy image and $s_i$ is the clean image
	- we consider the clean image $s_i$ as the mean of the distribution and event that follows the poisson distribution is the noisy image $x_i$
- Photons hitting each pixel is independent of other pixels
	- $p(\mathbf{x} \mid \mathbf{s}) = \prod_{i=1}^{n} p(x_i \mid s_i)$
	- allows us to represent the whole image as the product of each pixel poisson distribution
- A different way to represent pixels
	- $x_i=\sum_{t=1}^T\mathbb{1}(i_t=i)$
	- here $i = (i_1, i_2, ..., i_T)$ be the sequence of photons hitting an detector
	- where $i$ is the pixel location and $t$ be the photon location in the sequence
	- $t$ doesnt matter as they are independent
	- pixel intensity can be calculated by summing on the $i^{th}$ pixel position over the sequence
	- we can represent the image as the summation over the sequence of photons for each pixel location
- Considering the interval of sequence of photons, we can rewrite our equation for the image
	- $p({i}|s,T)=\begin{cases}\prod_{t=1}^Tp(i=i_t|s)&T=|{i}|\\0&T\neq |{i}|\end{cases}$
	- here $i$ in the LHS represents the image as the sequence of photons
	- the sequence of photons are represented by the pixel locations $i_t$, from previous point
	- Therefore, $i$ represents the image as a sequence of pixel locations.
	- $p({i}|s,T)=\begin{cases}\prod_{i=1}^Tp(i|s)^{x_i}&T=|{i}|\\0&T\neq |{i}|\end{cases}$
	- as $t$ : sequence position doesnt matter we rewrite it again as above
	- we additionally scale the probabilities of pixels according to their intensity
- Poisson distribution of each pixel
	- $p(i_t=i|s)=\frac{s_i}{\sum_{j=1}^ns_j}$
	- $s_i$ is the $i^{th}$ pixel of the clean image
	- here the ideal pixel probability is the normalized value of the clean image pixel $i$
- Rewrite the equation of the image again, to consider the absence of clean images
	- $p({i}|T)=\begin{cases}\prod_{t=1}^Tp(i=i_t|i_1,\ldots,i_{t-1},T)&T=|{i}|\\0&T\neq |{i}|\end{cases}$
	- here we write it as probability of seeing pixel ${i}$ next in the sequence of photons
- Loss function
	- $L(\theta) = -\sum_{k=1}^m \sum_{i=1}^n \ln f_i(x_{\text{inp}}^k; \theta) x_{\text{tar},i}^k$
	- $f_i$ is our model with parameters $\theta$, ${i}$ represents the pixel location
	- $x_{inp}^k$ is the input noisy image with photons removed $x_{tar,i}^k$ is the target photon locations
- Rewrite the Loss function to account for multiple photons as target
	- $L(\theta) = -\sum_{k=1}^m \frac{1}{n|x_{\text{tar}}^k|} \sum_{i=1}^n \ln f_i(x_{\text{inp}}^k; \theta) x_{\text{tar},i}^k$
	- since $x_{tar,i}^k$ is no more one-hot representation of the next photon location we need to normalize the equation
	- we divide the summation of pixels by ${n|x_{\text{tar}}^k|}$ to account for all the photon locations in the target


