# Capture webcam input and perform Fourier transformZ
import cv2
import numpy as np
from skimage import exposure

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

video_capture = cv2.VideoCapture(0)

while True:
	# Capture frame-by-frame
	ret, frame = video_capture.read()

	# Display the resulting frame
	cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
	imS = cv2.resize(frame, (400, 250))


	cv2.imshow('Original', imS[:,:,0])

	# Do the FFR of one of the image slices (grayscale)
	F = np.fft.fft2(imS[:,:,0])

	# Define the magnitude
	F_mag = 20*np.log(np.abs(np.fft.fftshift(F)))
	# Change it to 256 bits for display
	F_mag_bit = (((np.log(F_mag) - np.log(F_mag).min()) / (np.log(F_mag).max() - np.log(F_mag).min())) * 255.9).astype(np.uint8)
	#F_mag_bit_contrast = exposure.rescale_intensity(F_mag_bit, in_range=(100, 255))
	cv2.imshow('Mag', F_mag_bit)

	# Do the same for the phase
	F_phase = np.angle(np.fft.fftshift(F))
	F_phase_bit = (((F_phase - F_phase.min()) / (F_phase.max() - F_phase.min())) * 255.9).astype(np.uint8)
	F_phase_bit_contrast = exposure.rescale_intensity(F_phase_bit, in_range=(100, 255))

	cv2.imshow('Phase', F_phase_bit_contrast)


	if cv2.waitKey(1) & 0xFF == ord('q'):
		break



# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

'''
f=imS[:,:,0]
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
'''