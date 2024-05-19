
import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_highpass_filter(shape, sigma):
    
    kernel = np.zeros(shape, dtype=np.float32)
    center = shape[0] // 2
    for i in range(shape[0]):
        for j in range(shape[1]):
            distance = (i - center) ** 2 + (j - center) ** 2
            kernel[i, j] = np.exp(-distance / (2 * sigma ** 2))
    
    # Create a high-pass kernel by subtracting the Gaussian kernel from 1
    highpass_kernel = 1 - kernel
    
    return highpass_kernel

def dft_and_display(image_path, sigma):
    """
    Applies Gaussian high-pass filter to an image in the frequency domain
    and displays the original image, magnitude spectrum, phase spectrum,
    Gaussian high-pass filter, filtered magnitude spectrum, and filtered output.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    
    # Compute the Discrete Fourier Transform (DFT)
    dft_shift = np.fft.fftshift(np.fft.fft2(img))
    
    # Calculate magnitude and phase spectra
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    phase_spectrum = np.angle(dft_shift)
    
    # Create Gaussian high-pass filter
    highpass_filter = gaussian_highpass_filter(img.shape, sigma)
    
    # Apply Gaussian high-pass filter to the magnitude spectrum
    filtered_magnitude_spectrum = magnitude_spectrum * highpass_filter
    
    # Convert the filtered magnitude spectrum to spatial domain
    filtered_output = filtered_magnitude_spectrum * np.exp(1j * phase_spectrum)
    filtered_output = np.fft.ifft2(np.fft.ifftshift(filtered_output)).real
    
    # Plot images in a 2x3 grid
    plt.figure(figsize=(15, 10))
    titles = ['Input Image', 'Magnitude Spectrum', 'Phase Spectrum',
              'Gaussian High Pass Filter (Frequency Domain)', 'Filtered Magnitude Spectrum', 'Filtered Output']
    images = [img, magnitude_spectrum, phase_spectrum,
              np.log(np.abs(highpass_filter) + 1), filtered_magnitude_spectrum, filtered_output]
    cmap = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray']
    
    for i in range(6):
        plt.subplot(2, 3, i + 1), plt.imshow(images[i], cmap=cmap[i])
        plt.title(titles[i]), plt.xticks([]), plt.yticks([])
        plt.tick_params(labelsize=8)  # Set font size
    
    plt.tight_layout()
    plt.show()

# Process images with the Gaussian high-pass filter
print("Processing images with Gaussian High-pass Filter...")
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg",
               "image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg", "image10.jpg"]
for image_path in image_paths:
    print("Processing:", image_path)
    dft_and_display(image_path, sigma=30)  