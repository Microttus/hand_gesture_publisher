import time

import cv2
import glob
import numpy as np

# Set the camera index
camera_index = 0

# Open the camera
cap = cv2.VideoCapture(camera_index)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Set the resolution (optional, adjust as needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Calibration pattern size (adjust based on your calibration pattern)
pattern_size = (8, 6)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Load calibration images using glob
images = glob.glob('../images/*.jpg')  # Adjust the path pattern as needed

# Loop over calibration images
for fname in images:
    # Read the image
    img = cv2.imread(fname)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #cv2.imshow('img', gray)

    # Find corners in the image
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE)

    print(fname)
    # If corners are found, add object points and image points
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

        # Draw and display the corners (optional)
        img = cv2.drawChessboardCorners(img, pattern_size, corners, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

# Close the window (optional)
cv2.destroyAllWindows()



# Check if calibration images are found
if len(obj_points) > 0:
    # Calibrate the camera
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Print results
    if ret:
        print("Camera Matrix:")
        print(camera_matrix)
        print("\nDistortion Coefficients:")
        print(dist_coeffs)
    else:
        print("Error: Calibration failed.")
else:
    print("Error: No valid calibration images found.")

# Release the camera
cap.release()
