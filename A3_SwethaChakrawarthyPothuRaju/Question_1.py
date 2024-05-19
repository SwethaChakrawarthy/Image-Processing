import cv2
import numpy as np
import matplotlib.pyplot as plt

def dft_and_display(img_path, name):
    #read the image
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # Resizing the image
    img = cv2.resize(img, (400, 400), interpolation=cv2.INTER_AREA)
    
    # Computing the Discrete Fourier Transform (DFT)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    
    # Calculate magnitude spectrum and phase spectrum
    magnitude_spectrum = np.abs(dft_shift)
    phase_spectrum = np.angle(dft_shift)
    
    # Reconstruct images using magnitude and phase
    phase_shifted = dft_shift / magnitude_spectrum
    magnitude_only_idft = np.fft.ifftshift(magnitude_spectrum)
    magnitude_only_idft = np.fft.ifft2(magnitude_only_idft).real
    phase_only_idft = np.fft.ifftshift(phase_shifted)
    phase_only_idft = np.fft.ifft2(phase_only_idft).real
    reconstructed_image = magnitude_spectrum * np.exp(1j * phase_spectrum)
    reconstructed_image = np.fft.ifftshift(reconstructed_image)
    reconstructed_image = np.fft.ifft2(reconstructed_image).real
    
# Plot images with titles
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle(name, fontsize=16)
    
    titles = ['Input Image(a)', 'Magnitude Spectrum(b)', 'Phase Spectrum(c)', 
              'Reconstructed using only magnitude(d)', 'Reconstructed using only phase(e)', 'Reconstructed using both magnitude and phase(f)']
    images = [img, 20 * np.log1p(magnitude_spectrum), phase_spectrum,
              cv2.equalizeHist(cv2.normalize(magnitude_only_idft, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)),
              cv2.equalizeHist(cv2.normalize(phase_only_idft, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)),
              cv2.normalize(reconstructed_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)]
    cmap = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray']
    
    for i in range(2):
        for j in range(3):
            axes[i, j].imshow(images[i * 3 + j], cmap=cmap[i * 3 + j])
            axes[i, j].set_title(titles[i * 3 + j])
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()
# Paths to skin tumor and skin disease images
skin_tumor_images = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg"]
skin_disease_images = ["image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg"]

# Process every skin tumor and display the final outputs
print("Skin Tumor Image")
for image_path in skin_tumor_images:
    print("Processing output:", image_path)
    dft_and_display(image_path,"Tumor Image")

# Process every skin disease and display the final outputs
print("Skin Disease Image")
for image_path in skin_disease_images:
    print("Processing output:", image_path)
    dft_and_display(image_path,"Disease Image")


