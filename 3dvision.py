import cv2
import open3d as o3d
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

global_images_data = []
camera_poses = []


directory = 'Dog_RGB'
pattern = re.compile(r'im_\d{4}\.JPG')
image_filenames = [os.path.join(directory, f) for f in os.listdir(directory) if pattern.match(f)]
image_filenames.sort()  # Sort the filenames in ascending order
index = 0

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

###############################################################

def compute_sift(image):
    """
    Compute SIFT keypoints and descriptors for the input image.

    Parameters:
        image (numpy.ndarray): Input image.

    Returns:
        keypoints (list): List of keypoint objects.
        descriptors (numpy.ndarray): Descriptors for the corresponding keypoints.
    """
    # Convert the image to grayscale if it is not already
    if image.ndim == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Initialize the SIFT detector object
    sift = cv2.SIFT_create()
    
    # Compute the keypoints and descriptors
    keypoints, descriptors = sift.detectAndCompute(gray_image, None)

    return keypoints, descriptors

#########################3

def get_camera_pose(E, pts1, pts2, K):
    """
    Get the rotation and translation from the Essential Matrix.
    R and t give the transformation from the second image to the first.
    """
    _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
    return R, t

###################################

def feature_matching(descriptors1, descriptors2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    return good_matches


##################

def estimate_relative_pose(matches, keypoints1, keypoints2, K):
    src_pts = np.float32([ keypoints1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
    dst_pts = np.float32([ keypoints2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)
    
    E, mask = cv2.findEssentialMat(src_pts, dst_pts, K)
    _, R, t, mask = cv2.recoverPose(E, src_pts, dst_pts, K)
    
    return R, t

########################3

def triangulate_points(projMat1, projMat2, keypoints1, keypoints2, matches):
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
    
    points_homogeneous = cv2.triangulatePoints(projMat1, projMat2, points1.T, points2.T).T
    points_3D = points_homogeneous[:, :3] / points_homogeneous[:, 3, np.newaxis]
    
    return points_3D


##########################################################################3

def add_image_data(image):
    """
    Add image data to the global_images_data list.

    Parameters:
        image_path (str): Path to the image file.
    """
    # Compute SIFT keypoints and descriptors
    keypoints, descriptors = compute_sift(image)

    # Add data to the global variable
    global_images_data.append([image, keypoints, descriptors])

########################3

def draw_matches(image1, keypoints1, image2, keypoints2, matches):
    out_img = cv2.drawMatches(image1, keypoints1, image2, keypoints2, matches[:100], None, flags=2)
    cv2.imshow('Matches', out_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

###################

def get_sift_matches(kp1, des1, kp2, des2):
    """
    Extracts SIFT features and performs feature matching using BFMatcher.
    """
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

#################

if __name__ == '__main__':
    for i in range(len(image_filenames)-1):
        image = load_and_resize_image(image_filenames[i])
        add_image_data(image)
        print(len(global_images_data))
    print("done!")

    for i in range(len(global_images_data)-1):
        image1, keypoints1, descriptors1 = global_images_data[i]
        image2, keypoints2, descriptors2 = global_images_data[i+1]

        # Feature matching
        pts1, pts2 = get_sift_matches(keypoints1, descriptors1, keypoints2, descriptors2)

        # Estimating relative pose between image1 and image2
        E, _ = cv2.findEssentialMat(pts2, pts1, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        R, t = get_camera_pose(E, pts1, pts2, K)

        # If it's the first image pair, initialize the first camera pose
        if i == 0:
            camera_poses.append((np.eye(3), np.zeros((3, 1))))  # Append the initial camera pose
        # Accumulate camera poses
        R_previous, t_previous = camera_poses[-1]  # Get the previous pose
        R_current = R_previous @ R  # Update rotation
        t_current = t_previous + t  # Update translation, not matrix multiplication
        camera_poses.append((R_current, t_current))

    # Plotting the camera poses
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting camera poses
    for R, t in camera_poses:
        ax.scatter(t[0], t[1], t[2], marker='o')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Show plot
    plt.show()



