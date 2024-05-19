import cv2
import numpy as np
import matplotlib.pyplot as plt

def ideal_low_pass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    
    # Create a mask for ideal low-pass filter
    mask = np.zeros((rows, cols, 2), np.uint8)
    for i in range(rows):
        for j in range(cols):
            if (i - crow)**2 + (j - ccol)**2 <= cutoff**2:
                mask[i, j] = 1
    
    return mask

def ideal_high_pass_filter(shape, cutoff):
    # Create an ideal low-pass filter
    ideal_lp_filter = ideal_low_pass_filter(shape, cutoff)
    
    # Create an ideal high-pass filter by subtracting the low-pass filter from a matrix of ones
    ideal_hp_filter = np.ones(ideal_lp_filter.shape, np.uint8) - ideal_lp_filter
    
    return ideal_hp_filter

def dft_and_display(image_path, cutoff_lp, cutoff_hp):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    
    # Compute the Discrete Fourier Transform (DFT)
    dft_shift = np.fft.fftshift(np.fft.fft2(img))
    
    # Calculate magnitude and phase spectra
    magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
    phase_spectrum = np.angle(dft_shift)
    
    # Ideal low-pass filter
    ideal_lp_filter = ideal_low_pass_filter(img.shape, cutoff_lp)
    
    # Ensure dimensions match for multiplication
    ideal_lp_filter = ideal_lp_filter[:, :, 0]
    
    # Apply ideal low-pass filter
    dft_shift_filtered_lp = dft_shift * ideal_lp_filter
    
    # Reconstruct the image after low-pass filtering
    img_back_lp = np.fft.ifft2(np.fft.ifftshift(dft_shift_filtered_lp))
    img_back_lp = np.abs(img_back_lp)
    img_back_lp = cv2.normalize(img_back_lp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Ideal high-pass filter
    ideal_hp_filter = ideal_high_pass_filter(img.shape, cutoff_hp)
    
    # Ensure dimensions match for multiplication
    ideal_hp_filter = ideal_hp_filter[:, :, 0]
    
    # Apply ideal high-pass filter
    dft_shift_filtered_hp = dft_shift * ideal_hp_filter
    
    # Reconstruct the image after high-pass filtering
    img_back_hp = np.fft.ifft2(np.fft.ifftshift(dft_shift_filtered_hp))
    img_back_hp = np.abs(img_back_hp)
    img_back_hp = cv2.normalize(img_back_hp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Plot images in a 3x3 grid
    plt.figure(figsize=(15, 15))
    titles = ['Input Image', 'Magnitude Spectrum', 'Phase Spectrum',
              'Ideal Low-Pass Filter', 'Filtered Magnitude (LP)', 'Low-Pass Filtered Image',
              'Ideal High-Pass Filter', 'Filtered Magnitude (HP)', 'High-Pass Filtered Image']
    images = [img, magnitude_spectrum, phase_spectrum,
              20 * np.log(ideal_lp_filter + 1), magnitude_spectrum * ideal_lp_filter,
              img_back_lp,
              20 * np.log(ideal_hp_filter + 1), magnitude_spectrum * ideal_hp_filter,
              img_back_hp]
    cmap = ['gray', 'gray', 'gray', 'gray', 'gray', 'gray',
             'gray', 'gray', 'gray']
    
    for i in range(9):
        plt.subplot(2, 6, i + 1), plt.imshow(images[i], cmap=cmap[i])
        plt.title(titles[i]), plt.xticks([]), plt.yticks([])
        plt.tick_params(labelsize=8)  # Set font size
    
    plt.tight_layout()
    plt.show()


# Process images with both ideal low-pass and high-pass filters
print("Processing images with Ideal Low-Pass and High-Pass Filters...")
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg",
               "image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg", "image10.jpg"]
for image_path in image_paths:
    print("Processing:", image_path)
    dft_and_display(image_path, 30, 30)  

