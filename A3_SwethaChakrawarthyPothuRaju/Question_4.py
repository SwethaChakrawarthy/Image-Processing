import cv2
import numpy as np
import matplotlib.pyplot as plt

def feature_extraction(image,num):
    # Resize the image
    img = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocessing: Gaussian Blur
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)

    # Canny Edge Detection
    if num < 5:
        edges = cv2.Canny(img_blur, 150, 250)
    else:  
        edges= cv2.Canny(img_blur, 10, 250)

    # SIFT (Scale-Invariant Feature Transform) Feature Extraction
    sift = cv2.SIFT_create()
    keypoints_sift, descriptors_sift = sift.detectAndCompute(img_blur, None)

    # Create a canvas for SIFT keypoints
    img_with_sift = cv2.cvtColor(img_blur, cv2.COLOR_GRAY2BGR)

    # Draw SIFT keypoints on the canvas
    cv2.drawKeypoints(img_blur, keypoints_sift, img_with_sift, color=(0, 255, 0))

    # Plot images using Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(10, 10))

    # Original Image and Canny Edges with SIFT Mask
    axes[0].imshow(img_blur, cmap='gray')
    axes[0].set_title('Original Image')

    # SIFT Keypoints and Masked Canny Edges
    axes[1].imshow(img_with_sift)
    axes[1].set_title('SIFT Keypoints')
    axes[2].imshow(edges, cmap='gray')
    axes[2].set_title('Canny Edges')

    # Hide axes
    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Process images with the ideal band-pass filter
print("Processing the feature extraction...")
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg",
               "image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg", "image10.jpg"]
for i in range(0,10):
    print("Processing:", image_paths[i])
    img = cv2.imread(image_paths[i])  # Load the image
    feature_extraction(img,i)  # Pass the image to the function
