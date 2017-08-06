# two dimensional FFT example
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import math

# get the figure
f = plt.imread("Images/Dara2017.jpg")
# hack to make images with more than 2-dimensions work
if len(f.shape) > 2:
	f = f[:,:,0]

# Smooth the edges
x, y = np.meshgrid(np.linspace(-1,1,f.shape[1]), np.linspace(-1,1,f.shape[0]))
d = np.sqrt(x*x+y*y)
sigma, mu = 0.5, 0.0
g = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
f2 = (f * g)

# Two dimension FFT -- F is complex consisting of an amplitude and phase
F = np.fft.fft2(f2)


# find the amp and phase -- shift to put 0 wavenumber at the center
F_mag = np.abs(np.fft.fftshift(F))
F_phase = np.angle(np.fft.fftshift(F))

#Produce a plot of the original image, amplitude and phase
plt.rc("font", size=10)

plt.subplot(131)
plt.imshow(f, cmap=matplotlib.cm.Greys_r)
plt.title("original image")

plt.subplot(132)
plt.imshow(np.log(F_mag))
plt.title("|F(k)|")

plt.subplot(133)
plt.imshow(F_phase)
plt.title("phase of F(k)")

f = plt.gcf()
f.set_size_inches(10.0,6.0)

plt.savefig("fft2d.png", bbox_inches="tight")
plt.clf()


#-------------------------------------------------------------------------------
# scramble phase
Fnew_phase = 2.0*math.pi*np.random.rand(F_phase.shape[0], F_phase.shape[1])

# back to the complex representation
Fnew = F_mag*np.exp(1j*Fnew_phase)

fnew = np.fft.ifft2(np.fft.ifftshift(Fnew))

plt.imshow(np.real(fnew), cmap=matplotlib.cm.Greys_r)
plt.title(r"F$^{-1}$(F(k)) with scrampled phases")
plt.savefig("fft2d_phasescamble.png", bbox_inches="tight")

plt.clf()

#-------------------------------------------------------------------------------
# scramble amplitude
Fnew_mag = np.max(F_mag)*np.random.rand(F_mag.shape[0], F_mag.shape[1])

# back to the complex representation
Fnew = Fnew_mag*np.exp(1j*F_phase)

fnew = np.fft.ifft2(np.fft.ifftshift(Fnew))

plt.imshow(np.real(fnew), cmap=matplotlib.cm.Greys_r)
plt.title(r"F$^{-1}$(F(k)) with scrampled amplitudes")
plt.savefig("fft2d_magscamble.png", bbox_inches="tight")
plt.clf()


#-------------------------------------------------------------------------------
# filter out high and low spatial frequencies
# http://glowingpython.blogspot.com/2011/08/fourier-transforms-and-image-filtering.html
f = plt.imread("Images/Dara2017.jpg")
if len(f.shape) > 2:
	f = f[:,:,0]

F_orig = np.fft.fftshift(F)

P = np.zeros(F.shape, dtype=np.complex128)
Q = np.zeros(F.shape, dtype=np.complex128)

# Define a circle which will be used for masking
frac = 0.1
rad = frac*int(min(F.shape)/2)


ic = F.shape[0]/2
jc = F.shape[1]/2

# Loop through the image pixels
# Fix this so it is faster and uses the masking feature of numpy
for i in range(F.shape[0]):
	for j in range(F.shape[1]):
		# if the pixel is greater than the defined radius keep it zero.
		if math.sqrt( (i-ic)**2 + (j-jc)**2) < rad:
			P[i,j] = F_orig[i,j]
		# if the pixel is less than the defined radius keep it zero
		if math.sqrt( (i-ic)**2 + (j-jc)**2) > rad:
			Q[i,j] = F_orig[i,j]

# Fourier transform the filtered images
f_filtered_P = np.real(np.fft.ifft2(np.fft.ifftshift(P)))
f_filtered_Q = np.real(np.fft.ifft2(np.fft.ifftshift(Q)))

plt.subplot(321)
plt.imshow(np.log(np.abs(F_orig)))
plt.title("original |F(k)|")

plt.subplot(322)
plt.imshow(f, cmap=matplotlib.cm.Greys_r)
plt.title("original image")

plt.subplot(323)
plt.imshow(np.log(np.abs(P)))
plt.title("filtered |F(k)|")

plt.subplot(324)
plt.imshow(f_filtered_P, cmap=matplotlib.cm.Greys_r)
plt.title(r"filtered F$^{-1}$(F(k))")

plt.subplot(325)
plt.imshow(np.log(np.abs(Q)))
plt.title("filtered |F(k)|")

plt.subplot(326)
plt.imshow(f_filtered_Q, cmap=matplotlib.cm.Greys_r)
plt.title(r"filtered F$^{-1}$(F(k))")

f = plt.gcf()
f.set_size_inches(12.0,12.0)

plt.savefig("fft2d_filtered.png", bbox_inches="tight")

#-------------------------------------------------------------------------------
# The effect of partial sampling
plt.clf()

f = plt.imread("Images/Dara2017.jpg")
if len(f.shape) > 2:
	f = f[:,:,0]

F_orig = np.fft.fftshift(F)

P_mag = np.zeros(F_mag.shape, dtype=np.complex128)
P_phase = np.zeros(F_phase.shape, dtype=np.complex128)

# Select a range of random points and set them to 0
Q = np.random.rand(F.shape[0],F.shape[1])

# Loop through the image pixels
for i in range(F.shape[0]):
	for j in range(F.shape[1]):
		if Q[i,j] > 0.1:
			P_mag[i-3:i+3,j-3:3+3] = 0.0
			P_phase[i-3:i+3,j-3:3+3] = 0.0
		else:
			P_mag[i,j] = F_mag[i,j]
			P_phase[i,j] = F_phase[i,j]

# Fourier transform the filtered images back into the x,y plane
Fnew = P_mag*np.exp(1j*P_phase)
f_filtered_P = np.fft.ifft2(np.fft.ifftshift(Fnew))

plt.subplot(221)
plt.imshow(np.log(np.abs(F_orig)))
plt.title("original |F(k)|")

plt.subplot(222)
plt.imshow(f, cmap=matplotlib.cm.Greys_r)
plt.title("original image")

plt.subplot(223)
plt.imshow(np.log(np.abs(P_mag)))
plt.title("filtered |F(k)|")

plt.subplot(224)
plt.imshow(np.real(f_filtered_P), cmap=matplotlib.cm.Greys_r)
plt.title(r"filtered F$^{-1}$(F(k))")

f = plt.gcf()
f.set_size_inches(12.0,12.0)

plt.savefig("fft2d_partial.png", bbox_inches="tight")