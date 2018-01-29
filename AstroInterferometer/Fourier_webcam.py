# Capture webcam input and perform Fourier transform
import cv2
import numpy as np
from skimage import exposure

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

# Define the size of the window
window_size = (400,250)

# Define a circle which will be used for masking
xx, yy = np.mgrid[:window_size[1], :window_size[0]]
circle = (xx - window_size[1]/2.0) ** 2 + (yy - window_size[0]/2.0) ** 2


while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	# Display the resulting frame
	cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.resize(gray, (window_size[0], window_size[1]))
	cv2.imshow('Original', gray)

	# Smooth the edges using a Gaussian
	x, y = np.meshgrid(np.linspace(-1, 1, 400), np.linspace(-1, 1, 250))
	d = np.sqrt(x * x + y * y)
	sigma, mu = 0.5, 0.0
	gauss = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
	f2 = (gray * gauss)

	# Do the FFT of one of the image slices
	F = np.fft.fft2(f2)

	# Define the magnitude
	F_mag = np.abs(np.fft.fftshift(F))
	# Change it to 256 bits for display
	F_mag_bit = (((np.log(F_mag) - np.log(F_mag).min()) / (np.log(F_mag).max() - np.log(F_mag).min())) * 255.9).astype(np.uint8)
	#F_mag_bit_contrast = exposure.rescale_intensity(F_mag_bit, in_range=(100, 255))
	cv2.imshow('Mag', F_mag_bit)

	# Do the same for the phase
	F_phase = np.angle(np.fft.fftshift(F))
	F_phase_bit = (((F_phase - F_phase.min()) / (F_phase.max() - F_phase.min())) * 255.9).astype(np.uint8)
	F_phase_bit_contrast = exposure.rescale_intensity(F_phase_bit, in_range=(100, 255))
	F_phase_bit_contrast[circle > 30000] = 0.0

	cv2.imshow('Phase', F_phase_bit_contrast)

	F_low = np.zeros(F.shape, dtype=np.complex128)

	# Now mask the phase and magnitude images so we can fft them back and see the image
	F_low[circle < 30000] = F[circle < 30000]
	#F_high[circle > 30000] = F[circle > 30000]

	# Fourier transform the filtered images back to the image plane
	f_filtered_low = np.real(np.fft.ifft2(np.fft.ifftshift(F_low)))
	#f_filtered_high = np.real(np.fft.ifft2(np.fft.ifftshift(F_high)))

	cv2.imshow('low', np.log(np.abs(f_filtered_low)))


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
