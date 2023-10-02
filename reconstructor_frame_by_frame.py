import cv2
import open3d as o3d
import os
import re
import numpy as np



directory = 'Dog_RGB'

# Use a regular expression to match filenames of the form 'im_XXXX.jpg'
pattern = re.compile(r'im_\d{4}\.JPG')

# Get the list of all files that match the pattern in the directory
image_filenames = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
image_filenames.sort()  # Sort the filenames in ascending order

print(len(image_filenames))
# Handle the case when there are no matching images in the directory
if not image_filenames:
    raise ValueError(f"No matching images found in directory: {directory}")

# Initialize the index to load the first image
index = 0

def load_and_resize_image(file_path):
    # Load the image
    image = cv2.imread(file_path)
    
    # Resize the image to 1/4 of its original size
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (width // 4, height // 4))  # Reduce both width and height by half
    
    return resized_image

# Load the first image and create an empty image of the same size
current_image = load_and_resize_image(image_filenames[index])
height, width = current_image.shape[:2]
empty_image = np.zeros((height, width, 3), dtype=np.uint8)  # Create an empty (black) image

margin_width = 10
margin = np.zeros((height, margin_width, 3), dtype=np.uint8)

# Display the empty image on the left and the current image on the right
concatenated_image = cv2.hconcat([empty_image, margin ,current_image])
cv2.imshow('Image', concatenated_image)

# Initialize ORB detector
orb = cv2.ORB_create()

def display_images_with_features(left_img, right_img):
    # Find the keypoints and descriptors with ORB for both images
    keypoints_left, descriptors_left = orb.detectAndCompute(left_img, None)
    keypoints_right, descriptors_right = orb.detectAndCompute(right_img, None)

    img_with_features_left = cv2.drawKeypoints(left_img, keypoints_left, outImage=None, color=(0, 255, 0))
    img_with_features_right = cv2.drawKeypoints(right_img, keypoints_right, outImage=None, color=(0, 255, 0))

    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    img_matches = cv2.drawMatches(img_with_features_left, keypoints_left, img_with_features_right, keypoints_right, matches, outImg=None, flags=2)


    # Draw the keypoints on both images
    cv2.imshow('Image', img_matches)

display_images_with_features(empty_image, current_image)

vis = o3d.visualization.Visualizer()
vis.create_window('3D Visualization', width=640, height=480)

image_width_pixels = 1704  # Width of the image in pixels
image_height_pixels = 2272  # Height of the image in pixels, adjust as per your resized image
sensor_width_mm = 17.3  # Sensor width in mm
sensor_height_mm = 13.0
focal_length_mm = 34  # Focal length in mm

# Calculate the focal lengths in pixels
fx = (image_width_pixels / sensor_width_mm) * focal_length_mm
fy = (image_height_pixels / sensor_height_mm) * focal_length_mm

# Calculate the principal points
cx = image_width_pixels / 2
cy = image_height_pixels / 2

# Create the PinholeCameraIntrinsic object with the given parameters
camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(image_width_pixels, image_height_pixels, fx, fy, cx, cy)

while True:
    
    key = cv2.waitKey(0)
    if key == ord('n'):
        # If 'n' is pressed, load and display the next resized image if available
        index += 1
        print(index)
        if index < len(image_filenames):
            # Load and display new current image with features
            new_current_image = load_and_resize_image(image_filenames[index])
            display_images_with_features(current_image, new_current_image)

            # Update the current image
            current_image = new_current_image
        else:
            index = 0
            print("This is the last image.")
    elif key == 27:  # ASCII of ESC key
        break

vis.destroy_window()
cv2.destroyAllWindows()