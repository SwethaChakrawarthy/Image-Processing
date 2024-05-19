import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image:", image_path)
        return

    # Step 1: Enhance contrast
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv_image)
    v = cv2.equalizeHist(v)
    enhanced_hsv = cv2.merge([h, s, v])
    enhanced_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # Step 2: Apply a selective color filter
    lower_red = np.array([0, 120, 0])
    upper_red = np.array([10, 255, 255])
    mask1 = cv2.inRange(enhanced_hsv, lower_red, upper_red)
    lower_red = np.array([170, 120, 0])
    upper_red = np.array([180, 255, 255])
    mask2 = cv2.inRange(enhanced_hsv, lower_red, upper_red)
    combined_red_mask = mask1 | mask2

    # Step 3: Use morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    cleaned_mask = cv2.morphologyEx(combined_red_mask, cv2.MORPH_CLOSE, kernel)
    cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

    # Step 4: Apply the cleaned mask to the image and find edges
    masked_image = cv2.bitwise_and(enhanced_image, enhanced_image, mask=cleaned_mask)
    edges_final = cv2.Canny(masked_image, threshold1=100, threshold2=200)

    # Step 5: Refine the edges using contour analysis
    contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = image.copy()
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        epsilon = 0.02 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(contour_image, [approx], -1, (0, 255, 0), 2)

    # Initialize the ORB keypoint detector
    orb = cv2.ORB_create()
    keypoints_orb = orb.detect(masked_image, None)
    keypoints_orb, descriptors_orb = orb.compute(masked_image, keypoints_orb)
    orb_keypoint_image = cv2.drawKeypoints(image, keypoints_orb, None, color=(0, 255, 0), flags=0)

    # Plot the results
    plt.figure(figsize=(18, 12))
    plt.subplot(2, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(cv2.cvtColor(contour_image, cv2.COLOR_BGR2RGB))
    plt.title('Contour Image')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(cv2.cvtColor(orb_keypoint_image, cv2.COLOR_BGR2RGB))
    plt.title('ORB Keypoint Detection')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(edges_final, cmap='gray')
    plt.title('Edges Detected')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

    
if __name__ == "__main__":
    # List of image paths to process
    image_paths = ['steroid acne.jpg', 'steroid acne2.jpg']  # Update these paths as necessary
    for image_path in image_paths:
        process_image(image_path)
