# import cv2
# import numpy as np
#
# # Load the source image and the template image
# source_image = cv2.imread('source2.jpg', 0)  # Load as grayscale
# template_image = cv2.imread('template.jpg', 0)  # Load as grayscale
#
# # Get the dimensions of the template image
# template_height, template_width = template_image.shape
#
# # Perform template matching
# result = cv2.matchTemplate(source_image, template_image, cv2.TM_CCOEFF_NORMED)
#
# # Find the location with the highest match score
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#
# # Draw a rectangle around the best match
# top_left = max_loc
# bottom_right = (top_left[0] + template_width, top_left[1] + template_height)
# cv2.rectangle(source_image, top_left, bottom_right, 255, 2)
#
# # Show the result
# cv2.imshow('Matched Image', source_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

import cv2
import numpy as np

# Function to rotate an image and return the rotated image and new coordinates
def rotate_image_and_get_coordinates(image, angle):
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(rotation_matrix[0, 0])
    sin = np.abs(rotation_matrix[0, 1])

    # Compute the new bounding dimensions
    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    # Adjust the rotation matrix to account for translation
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]

    # Perform the actual rotation and return the image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height))
    return rotated_image, rotation_matrix

# Load the source and template images
source_image = cv2.imread('source.jpg', 0)
template_image = cv2.imread('template.jpg', 0)

# Variables to track the best match
best_match_score = -float('inf')
best_match_location = None
best_angle = 0
best_rotation_matrix = None

# Perform template matching for each rotated version of the template
for angle in range(-10, 11):  # Rotate from -10° to +10°
    rotated_template, rotation_matrix = rotate_image_and_get_coordinates(template_image, angle)
    result = cv2.matchTemplate(source_image, rotated_template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val > best_match_score:
        best_match_score = max_val
        best_match_location = max_loc
        best_angle = angle
        best_rotation_matrix = rotation_matrix

# Draw the rotated bounding box on the source image
template_height, template_width = template_image.shape
top_left = best_match_location
points = np.array([
    [top_left[0], top_left[1]],
    [top_left[0] + template_width, top_left[1]],
    [top_left[0] + template_width, top_left[1] + template_height],
    [top_left[0], top_left[1] + template_height]
])

# Apply the best rotation matrix to the bounding box points
rotated_points = cv2.transform(np.array([points]), best_rotation_matrix)[0]

# Draw the rotated bounding box
rotated_image_color = cv2.cvtColor(source_image, cv2.COLOR_GRAY2BGR)
cv2.polylines(rotated_image_color, [np.int32(rotated_points)], isClosed=True, color=(0, 255, 0), thickness=2)

# Show the result
cv2.imshow('Best Rotated Match', rotated_image_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

