import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from skimage.feature import hog
from skimage import exposure

# Load the digits dataset from sklearn
digits = datasets.load_digits()
images = digits.images
labels = digits.target

# Display counts of each label
unique, counts = np.unique(labels, return_counts=True)
label_counts = dict(zip(unique, counts))
print("Label Counts:", label_counts)

# Number of images to display
num_images = 12  # Change this to display more/less images

# Set up the plot
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8))
axes = axes.flatten()

for i in range(num_images):
    axes[i].imshow(images[i], cmap='gray')
    axes[i].set_title(f'Label: {labels[i]}')
    axes[i].axis('off')

plt.tight_layout()
plt.show()

# Pre-processing example: Select a test image and process it
ex_test_image = images[37]  # Example test image
processed_image = (ex_test_image > 0.5).astype(np.float64)  # Binarize the grayscale image

plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(ex_test_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(processed_image, cmap='gray')
plt.title('Processed Image')
plt.axis('off')

plt.tight_layout()
plt.show()

# Check the shape of the selected image
img = images[206]  # Select an image for HOG feature extraction
print("Selected Image Shape:", img.shape)  # Print the shape

# Ensure the image is 2D before HOG extraction
if img.ndim == 2:
    # Extract HOG features and visualize with valid cell sizes
    hog_features_2x2, hog_image_2x2 = hog(img, pixels_per_cell=(2, 2), visualize=True)
    hog_features_4x4, hog_image_4x4 = hog(img, pixels_per_cell=(4, 4), visualize=True)
    
    # Do not use 8x8 for 8x8 images
    # hog_features_8x8, hog_image_8x8 = hog(img, pixels_per_cell=(8, 8), visualize=True)

    # Show the original image and HOG features
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(hog_image_2x2, cmap='gray')
    plt.title(f'HOG Cell Size = [2 2]\nLength = {len(hog_features_2x2)}')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(hog_image_4x4, cmap='gray')
    plt.title(f'HOG Cell Size = [4 4]\nLength = {len(hog_features_4x4)}')
    plt.axis('off')

    # If you want to visualize a larger cell size, consider changing the dataset
    # plt.subplot(2, 3, 6)
    # plt.imshow(hog_image_8x8, cmap='gray')
    # plt.title(f'HOG Cell Size = [8 8]\nLength = {len(hog_features_8x8)}')
    # plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Get the HOG feature size for cell size [4 4]
    cell_size = (4, 4)
    hog_feature_size = len(hog_features_4x4)
else:
    print("The selected image is not a 2D array. Check the input data.")
