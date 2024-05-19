import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_skin_lesion(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not open or find the image.")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Convert image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    
    # Segment the lesion using color thresholding in Lab color space
    lab_lower_bound = np.array([0, 140, 0])
    lab_upper_bound = np.array([255, 255, 255])
    lab_mask = cv2.inRange(lab_image, lab_lower_bound, lab_upper_bound)
    
    # Morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.erode(lab_mask, kernel, iterations=1)
    cleaned_mask = cv2.dilate(cleaned_mask, kernel, iterations=2)
    
    # Apply the cleaned mask to the grayscale image
    masked_gray = cv2.bitwise_and(blurred, blurred, mask=cleaned_mask)
    
    # Perform Canny edge detection
    edges = cv2.Canny(masked_gray, 100, 200)
    
    # Find contours on the cleaned mask
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on the original image for visualization
    contour_image = image.copy()
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Create a mask for keypoints
    keypoint_mask = np.zeros_like(gray)
    cv2.drawContours(keypoint_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Detect ORB features within the lesion only
    orb = cv2.ORB_create()
    key_points_masked, descriptors = orb.detectAndCompute(image, keypoint_mask)
    keypoint_image_masked = cv2.drawKeypoints(image, key_points_masked, None, color=(0, 255, 0), flags=0)
    
    return edges, contour_image, keypoint_image_masked, image

# List of image paths
image_paths = ['eczema.jpg', 'eczema2.jpg', 'guttate Psoriasis.jpg', 'guttate Psoriasis3.jpg']

# Process each image and display the results
for i, image_path in enumerate(image_paths, start=1):
    edges, contour_image, keypoint_image_masked, image = process_skin_lesion(image_path)
    
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f'Original Image {i}')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title(f'Contour Image {i}')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Edges Detected {i}')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(keypoint_image_masked, cv2.COLOR_BGR2RGB))
    plt.title(f'ORB Keypoints Image {i}')
    plt.axis('off')

    plt.show()
