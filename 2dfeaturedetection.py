import cv2
import open3d as o3d
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

global_3D_points = []
global_R = np.eye(3)
global_t = np.zeros((3, 1))
relative_R = []
relative_t = []
P1 = None
P2 = None

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

def plot_camera_trajectory_and_3d_points(relative_R, relative_t, all_3D_points):
    """
    Plot the camera trajectory and 3D points.

    Parameters:
        relative_R (list): List of relative rotation matrices.
        relative_t (list): List of relative translation vectors.
        all_3D_points (np.array): Nx3 array of 3D points.
    """
    # Compute Absolute Poses
    absolute_R = [np.eye(3)]
    absolute_t = [np.zeros((3, 1))]

    for i in range(1, len(relative_R)):
        Ri = absolute_R[i-1] @ relative_R[i]
        ti = absolute_t[i-1] + absolute_R[i-1] @ relative_t[i]
        absolute_R.append(Ri)
        absolute_t.append(ti)

    # Visualization
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot 3D points
    ax.scatter(all_3D_points[:,0], all_3D_points[:,1], all_3D_points[:,2], s=1, c='b', marker='o')

    # Plot camera centers
    for i in range(len(absolute_R)):
        Ci = -absolute_R[i].T @ absolute_t[i]
        ax.scatter(Ci[0], Ci[1], Ci[2], marker='x', color='red')
        if i > 0:
            Cprev = -absolute_R[i-1].T @ absolute_t[i-1]
            ax.plot([Cprev[0], Ci[0]], [Cprev[1], Ci[1]], [Cprev[2], Ci[2]], 'k-', color='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=135, azim=90)  # Adjust the viewing angle for better visualization
    plt.title("Camera Trajectory and 3D Points")
    plt.show()

def load_and_resize_image(file_path):
    # Load the image
    image = cv2.imread(file_path)
    
    # Resize the image to 1/4 of its original size
    height, width = image.shape[:2]
    resized_image = cv2.resize(image, (width // 4, height // 4))  # Reduce both width and height by half
    
    return resized_image

def triangulate_points(points1, points2, P1, P2):
    # Using cv2.triangulatePoints to get the 3D points
    homogenous_3D_points = cv2.triangulatePoints(P1, P2, points1.T, points2.T)
    
    # Converting homogenous coordinates to 3D points
    cartesian_3D_points = homogenous_3D_points / homogenous_3D_points[3]
    
    # Transpose to get the 3D points in a useful format
    cartesian_3D_points = cartesian_3D_points[:3].T
    
    return cartesian_3D_points


# Load the first image and create an empty image of the same size
current_image = load_and_resize_image(image_filenames[index])
height, width = current_image.shape[:2]
empty_image = load_and_resize_image(image_filenames[index-1])
#empty_image = np.zeros((height, width, 3), dtype=np.uint8)  # Create an empty (black) image

vis = o3d.visualization.Visualizer()
vis.create_window('3D Visualization', width=640, height=480)

margin_width = 10
margin = np.zeros((height, margin_width, 3), dtype=np.uint8)

# Initialize SIFT detector
sift = cv2.SIFT_create()

def visualize_3d_points(vis, points_3d):
    # Creating an Open3D PointCloud object
    pcd = o3d.geometry.PointCloud()
    
    # Converting numpy array to Open3D format for points
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    # Clearing previous points in the visualizer and adding new points
    vis.clear_geometries()
    vis.add_geometry(pcd)
    
    # Updating the visualizer window
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()

def display_images_with_features(left_img, right_img):
    global P1, P2
    # Find the keypoints and descriptors with SIFT for both images
    keypoints_left, descriptors_left = sift.detectAndCompute(left_img, None)
    keypoints_right, descriptors_right = sift.detectAndCompute(right_img, None)

    # You may draw the keypoints on the images using drawKeypoints
    img_with_features_left = cv2.drawKeypoints(left_img, keypoints_left, outImage=None, color=(0, 255, 0))
    img_with_features_right = cv2.drawKeypoints(right_img, keypoints_right, outImage=None, color=(0, 255, 0))

    # Create BFMatcher object with L2 norm (SIFT uses L2 norm)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors_left, descriptors_right)

    # Sort them in ascending order of distance
    matches = sorted(matches, key=lambda x: x.distance)

    R, t = estimate_camera_pose(matches, keypoints_left, keypoints_right, K)

    if(index != 2):
        P1 = P2
    else:
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([global_R, global_t])

    
    points_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches])
    points_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches])
    
    # Performing the triangulation
    points_3D = triangulate_points(points_left, points_right, P1, P2)

    global_3D_points.append(points_3D)

    all_points_3D = np.vstack(global_3D_points)

    visualize_3d_points(vis, all_points_3D)
    # Draw the matched keypoints on both images
    img_matches = cv2.drawMatches(img_with_features_left, keypoints_left, img_with_features_right, keypoints_right, matches, outImg=None, flags=2)
    
    cv2.imshow('Image', img_matches)

    return points_3D

def estimate_camera_pose(matches, keypoints_left, keypoints_right, K):
    global global_R, global_t, relative_R, relative_t
    # Extract matched keypoints
    points_left = np.float32([keypoints_left[m.queryIdx].pt for m in matches])
    points_right = np.float32([keypoints_right[m.trainIdx].pt for m in matches])

    # Estimate the essential matrix
    E, mask = cv2.findEssentialMat(points_left, points_right, cameraMatrix=K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

    # Recover the rotational and translational matrix
    _, R, t, _ = cv2.recoverPose(E, points_left, points_right, cameraMatrix=K, mask=mask)

    global_R = R @ global_R
    global_t = R @ global_t + t
    relative_R.append(R)
    relative_t.append(t)
    return R, t


display_images_with_features(empty_image, current_image)



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

            if index % 10 == 0:  # Adjust the condition as per your requirement
                all_3D_points = np.vstack(global_3D_points)
                plot_camera_trajectory_and_3d_points(relative_R, relative_t, all_3D_points)
            # Update the current image
            current_image = new_current_image
        else:
            index = 0
            print("This is the last image.")
    elif key == 27:  # ASCII of ESC key
        break

vis.destroy_window()
cv2.destroyAllWindows()