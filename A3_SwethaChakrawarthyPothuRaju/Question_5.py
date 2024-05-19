import cv2
import numpy as np
import matplotlib.pyplot as plt

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced

def remove_hair_with_inpainting(image, mask):
    # Inpaint the hair regions using Navier-Stokes based algorithm
    inpainted_image = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_NS)
    
    return inpainted_image

def detect_edges(image, threshold1=150, threshold2=250):
    # Detect edges using Canny edge detection
    edges = cv2.Canny(image, threshold1, threshold2)
    
    return edges

def detect_sift_keypoints(image):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Detect keypoints
    keypoints = sift.detect(image, None)
    
    return keypoints

# List of image paths
print("Processing the feature extraction...")
image_paths = ["image1.jpg", "image2.jpg", "image3.jpg", "image4.jpg", "image5.jpg",
               "image6.jpg", "image7.jpg", "image8.jpg", "image9.jpg", "image10.jpg"]

# Process each image
for image_path in image_paths:
    # Load image
    image = cv2.imread(image_path)
    
    # Step 1: Preprocess the image
    preprocessed_image = preprocess_image(image)

    # Step 2: Create a mask to segment the lesion area
    _, mask = cv2.threshold(preprocessed_image, 200, 255, cv2.THRESH_BINARY)

    # Step 3: Remove hair using inpainting
    inpainted_image = remove_hair_with_inpainting(image, mask)

    # Step 4: Detect edges on the inpainted image
    edges = detect_edges(inpainted_image)

    # Step 5: Detect SIFT keypoints on the inpainted image
    keypoints = detect_sift_keypoints(inpainted_image)

    # Draw keypoints on the inpainted image
    image_with_keypoints = cv2.drawKeypoints(inpainted_image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Display the results
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Input Image')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
    plt.title('Inpainted Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(edges, cmap='gray')
    plt.title('Detected Edges')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(image_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Image with SIFT Keypoints')
    plt.axis('off')

    plt.show()
