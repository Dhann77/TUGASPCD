import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import measure

# Function to process and display each step
def process_image(image_path):
    # Load the image
    RGB = cv2.imread(image_path)
    RGB = cv2.cvtColor(RGB, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
    plt.figure()
    plt.imshow(RGB)
    plt.title("Original RGB Image")
    plt.show()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(RGB, cv2.COLOR_RGB2GRAY)
    plt.figure()
    plt.imshow(gray, cmap='gray')
    plt.title("Grayscale Image")
    plt.show()

    # Step 2: Apply thresholding (using Otsu's method)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    plt.figure()
    plt.imshow(binary, cmap='gray')
    plt.title("Binary Image (Thresholded)")
    plt.show()

    # Step 3: Fill holes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled_image = np.zeros_like(binary)
    for contour in contours:
        cv2.drawContours(filled_image, [contour], -1, 255, thickness=cv2.FILLED)
    plt.figure()
    plt.imshow(filled_image, cmap='gray')
    plt.title("After Filling Holes")
    plt.show()

    # Step 4: Identify contours and calculate properties
    labeled_image = RGB.copy()
    num_labels, labels_im = cv2.connectedComponents(filled_image)
    properties = measure.regionprops(labels_im)

    # Store results for display
    centroids = []
    areas = []
    perimeters = []

    for i, prop in enumerate(properties):
        if i == 0:  # Skip the background
            continue

        # Centroid
        cX, cY = prop.centroid
        centroids.append((cX, cY))

        # Area and Perimeter
        area = prop.area
        perimeter = prop.perimeter
        areas.append(area)
        perimeters.append(perimeter)

        # Boundary of the object
        boundary = np.array(prop.coords)
        
        # Display the boundary and labels on the image
        labeled_image[boundary[:, 0], boundary[:, 1]] = [255, 255, 0]  # Yellow boundary
        
        # Label each object with number, area, perimeter, and centroid
        cv2.putText(labeled_image, f"Label = {i}", (int(cY), int(cX) - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(labeled_image, f"Area = {area}", (int(cY), int(cX)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(labeled_image, f"Perim = {perimeter:.2f}", (int(cY), int(cX) + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(labeled_image, f"X = {int(cX)}", (int(cY), int(cX) + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(labeled_image, f"Y = {int(cY)}", (int(cY), int(cX) + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Display labeled image
    plt.figure()
    plt.imshow(labeled_image)
    plt.title("Labeled Image with Object Properties")
    plt.show()

    # Print calculated properties
    for i, (centroid, area, perimeter) in enumerate(zip(centroids, areas, perimeters), start=1):
        print("===================================")
        print(f"Object number = {i}")
        print(f"Centroid = ({centroid[1]:.2f}, {centroid[0]:.2f})")
        print(f"Area = {area}")
        print(f"Perimeter = {perimeter:.2f}")

# Process the image
process_image('C:\\Users\\a516j\\OneDrive\\Pictures\\Screenshots\\Screenshot 2024-11-05 010516.png')
