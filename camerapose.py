import cv2
import open3d as o3d
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

directory = 'Dog_RGB'
pattern = re.compile(r'im_\d{4}\.JPG')
image_filenames = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
image_filenames.sort()  # Sort the filenames in ascending order
index = 2

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

K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0,  1]])

K = K / 4

def load_and_resize_image(file_path):
    # Load the image
    image = cv2.imread(file_path)
    
    # Resize the image to 1/4 of its original size
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (width // 4, height // 4))  # Reduce both width and height by half
    
    return resized_image

def get_camera_pose(E, pts1, pts2, K):
    """
    Get the rotation and translation from the Essential Matrix.
    R and t give the transformation from the second image to the first.
    """
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

def get_sift_matches(img1, img2):
    """
    Extracts SIFT features and performs feature matching using BFMatcher.
    """
    # Convert images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)
    
    # Use BFMatcher (BRUTEFORCE_HAMMING) to find the best matches
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply ratio test
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append((m.trainIdx, m.queryIdx))
    
    # Get coordinates of the matched points
    pts1 = np.float32([kp1[i].pt for (_, i) in good])
    pts2 = np.float32([kp2[i].pt for (i, _) in good])
    
    return pts1, pts2

Rs = [np.eye(3)]  # rotation from world frame to camera frame
ts = [np.zeros((3, 1))]  # translation from world frame to camera frame

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


for i in range(len(image_filenames)-38):
    # Load and resize the images
    prev_img = load_and_resize_image(image_filenames[i])
    next_img = load_and_resize_image(image_filenames[i + 1])
    
    # Get matching points using SIFT
    pts1, pts2 = get_sift_matches(prev_img, next_img)
    
    # Estimate Essential Matrix
    E, _ = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
    
    # Estimate pose from Essential Matrix
    R, t = get_camera_pose(E, pts1, pts2, K)

    # Update pose
    Rs.append(R @ Rs[-1])
    ts.append(ts[-1] + Rs[-1] @ t)

    # Plot camera center (you may need to adjust scaling/units)
    ax.scatter(ts[-1][0], ts[-1][1], ts[-1][2], c='b', marker='o')
    ax.plot([ts[-2][0], ts[-1][0]], [ts[-2][1], ts[-1][1]], [ts[-2][2], ts[-1][2]], c='r')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()