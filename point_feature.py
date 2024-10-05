import cv2
import numpy as np
from matplotlib import pyplot as plt


#install these
'''
pip install opencv-python==3.4.2.16
pip install opencv-contrib-python==3.4.2.16 '''


# Read the images (Ensure that your images are in the correct path)
box_image = cv2.imread(r"C:\Users\Vinish\Downloads\CV_CAT\object_image.png", cv2.IMREAD_GRAYSCALE)
scene_image = cv2.imread(r"C:\Users\Vinish\Downloads\CV_CAT\cluttered_image.png", cv2.IMREAD_GRAYSCALE)

# Read the elephant image
elephant_image = cv2.imread(r"C:\Users\Vinish\Downloads\CV_CAT\ele.png", cv2.IMREAD_GRAYSCALE)

# Initialize the SURF detector (Hessian Threshold set to 400)
surf = cv2.xfeatures2d.SURF_create(400)  # You can adjust this value

# Detect keypoints and descriptors in the box image
box_keypoints, box_descriptors = surf.detectAndCompute(box_image, None)

# Detect keypoints and descriptors in the scene image
scene_keypoints, scene_descriptors = surf.detectAndCompute(scene_image, None)

# Detect keypoints and descriptors in the elephant image
elephant_keypoints, elephant_descriptors = surf.detectAndCompute(elephant_image, None)

# Draw the 100 strongest keypoints for the box image
box_keypoints = sorted(box_keypoints, key=lambda x: -x.response)[:100]
box_image_with_keypoints = cv2.drawKeypoints(box_image, box_keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw the 300 strongest keypoints for the scene image
scene_keypoints = sorted(scene_keypoints, key=lambda x: -x.response)[:300]
scene_image_with_keypoints = cv2.drawKeypoints(scene_image, scene_keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Draw the keypoints for the elephant image
elephant_keypoints = sorted(elephant_keypoints, key=lambda x: -x.response)[:100]
elephant_image_with_keypoints = cv2.drawKeypoints(elephant_image, elephant_keypoints, None, (0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Use Brute-Force matcher to match the features for box and scene
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches = bf.match(box_descriptors, scene_descriptors)

# Sort matches based on the distance (best matches first)
matches = sorted(matches, key=lambda x: x.distance)

# Filter valid matches by ensuring index boundaries
valid_matches = [m for m in matches if m.queryIdx < len(box_keypoints) and m.trainIdx < len(scene_keypoints)]

# Draw the matches (including outliers)
matched_box_points = np.float32([box_keypoints[m.queryIdx].pt for m in valid_matches])
matched_scene_points = np.float32([scene_keypoints[m.trainIdx].pt for m in valid_matches])

# Estimate the affine transformation using inliers
matched_box_points = np.float32([box_keypoints[m.queryIdx].pt for m in valid_matches]).reshape(-1, 1, 2)
matched_scene_points = np.float32([scene_keypoints[m.trainIdx].pt for m in valid_matches]).reshape(-1, 1, 2)

# Estimate affine transformation
tform, inliers = cv2.estimateAffinePartial2D(matched_box_points, matched_scene_points)

# Select inliers
inlier_box_points = matched_box_points[inliers.ravel() == 1]
inlier_scene_points = matched_scene_points[inliers.ravel() == 1]

# Draw matches with inliers only
inlier_matches = [m for i, m in enumerate(valid_matches) if inliers[i]]

# Prepare match images
match_img = cv2.drawMatches(box_image, box_keypoints, scene_image, scene_keypoints, valid_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
inlier_match_img = cv2.drawMatches(box_image, box_keypoints, scene_image, scene_keypoints, inlier_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Define the polygon representing the box corners in the box image
box_polygon = np.float32([[0, 0], 
                          [box_image.shape[1], 0], 
                          [box_image.shape[1], box_image.shape[0]], 
                          [0, box_image.shape[0]], 
                          [0, 0]]).reshape(-1, 1, 2)

# Transform the box polygon to the scene using the affine transform
new_box_polygon = cv2.transform(box_polygon, tform)

# Prepare the final image with the detected box
detected_box_image = scene_image.copy()
if tform is not None:
    cv2.polylines(detected_box_image, [np.int32(new_box_polygon)], isClosed=True, color=(255, 255, 0), thickness=2)

# Now repeat the detection for the elephant image
# Use Brute-Force matcher to match the features for elephant and scene
elephant_matches = bf.match(elephant_descriptors, scene_descriptors)

# Sort matches based on the distance (best matches first)
elephant_matches = sorted(elephant_matches, key=lambda x: x.distance)

# Filter valid matches for elephant
valid_elephant_matches = [m for m in elephant_matches if m.queryIdx < len(elephant_keypoints) and m.trainIdx < len(scene_keypoints)]

# Draw the matches for elephant (including outliers)
matched_elephant_points = np.float32([elephant_keypoints[m.queryIdx].pt for m in valid_elephant_matches]).reshape(-1, 1, 2)
matched_scene_elephant_points = np.float32([scene_keypoints[m.trainIdx].pt for m in valid_elephant_matches]).reshape(-1, 1, 2)

# Estimate the affine transformation using inliers for elephant
tform_elephant, inliers_elephant = cv2.estimateAffinePartial2D(matched_elephant_points, matched_scene_elephant_points)

# Draw matches with inliers only for elephant
inlier_elephant_matches = [m for i, m in enumerate(valid_elephant_matches) if inliers_elephant[i]]

# Prepare match images for elephant
elephant_match_img = cv2.drawMatches(elephant_image, elephant_keypoints, scene_image, scene_keypoints, valid_elephant_matches[:50], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
inlier_elephant_match_img = cv2.drawMatches(elephant_image, elephant_keypoints, scene_image, scene_keypoints, inlier_elephant_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Define the polygon representing the elephant corners in the elephant image
elephant_polygon = np.float32([[0, 0], 
                               [elephant_image.shape[1], 0], 
                               [elephant_image.shape[1], elephant_image.shape[0]], 
                               [0, elephant_image.shape[0]], 
                               [0, 0]]).reshape(-1, 1, 2)

# Transform the elephant polygon to the scene using the affine transform
new_elephant_polygon = cv2.transform(elephant_polygon, tform_elephant)

# Prepare the final image with the detected elephant
detected_elephant_image = scene_image.copy()
if tform_elephant is not None:
    cv2.polylines(detected_elephant_image, [np.int32(new_elephant_polygon)], isClosed=True, color=(0, 255, 255), thickness=2)

# Prepare the final image with both the detected box and elephant
final_detected_image = scene_image.copy()

# Draw the detected box polygon if transformation exists
if tform is not None:
    cv2.polylines(final_detected_image, [np.int32(new_box_polygon)], isClosed=True, color=(255, 255, 0), thickness=2)

# Draw the detected elephant polygon if transformation exists
if tform_elephant is not None:
    cv2.polylines(final_detected_image, [np.int32(new_elephant_polygon)], isClosed=True, color=(0, 255, 255), thickness=2)



# Plot the results in a tabular form
plt.figure(figsize=(10,10))

# Create a 4x4 subplot grid to include original images
plt.subplot(4, 3, 1)
plt.imshow(box_image, cmap='gray')
plt.title('Original Box Image')
plt.axis('off')

plt.subplot(4, 3, 2)
plt.imshow(scene_image, cmap='gray')
plt.title('Original Scene Image')
plt.axis('off')

plt.subplot(4, 3, 3)
plt.imshow(box_image_with_keypoints)
plt.title('Box Image Keypoints')
plt.axis('off')

plt.subplot(4, 3, 4)
plt.imshow(scene_image_with_keypoints)
plt.title('Scene Image Keypoints')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.imshow(match_img)
plt.title('Matched Points (Including Outliers)')
plt.axis('off')

plt.subplot(4, 3, 6)
plt.imshow(inlier_match_img)
plt.title('Matched Points (Inliers Only)')
plt.axis('off')

plt.subplot(4, 3, 7)
plt.imshow(detected_box_image, cmap='gray')
plt.title('Detected Box in Scene')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.imshow(elephant_image, cmap='gray')
plt.title('Original Elephant Image')
plt.axis('off')


plt.subplot(4, 3, 9)
plt.imshow(elephant_match_img)
plt.title('Matched Points (Including Outliers)')
plt.axis('off')



plt.subplot(4, 3, 10)
plt.imshow(inlier_elephant_match_img)
plt.title('Matched Points (Inliers Only)')
plt.axis('off')

plt.subplot(4, 3, 11)
plt.imshow(elephant_image_with_keypoints)
plt.title('Elephant Image Keypoints')
plt.axis('off')

plt.subplot(4, 3, 12)
plt.imshow(final_detected_image, cmap='gray')
plt.title('Detected Box and Elephant in Scene')
plt.axis('off')


plt.tight_layout()
plt.show()
