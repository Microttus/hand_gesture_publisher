# Oktma
# Hand palm location finder
# UiA Grimstad, 22.02.2024

# General pkg
import os
import numpy as np

# Image handling pkg
import cv2
import mediapipe as mp
import tensorflow as tf
import time
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

# ament pkg
#from ament_index_python.packages import get_package_share_directory

# ROS2 pkg
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

camera_index = 2

cap = cv2.VideoCapture(camera_index)
detector = HandDetector(maxHands=1)


class HandLocationFinder:

    def __init__(self):
        #package_share_direcroty = get_package_share_directory('hand_gesture_pub')
        #print(package_share_direcroty)
        #self.model = tf.keras.models.load_model(os.path.join(package_share_direcroty, 'CNN_LSTM_Latest.h5'))
        print("Camera index: {}".format(camera_index))

    def find_palm_pos(self):
        # print('Test function')

        success, img1 = cap.read()
        hand, img1 = detector.findHands(img1)

        try:
            palm_pos = hand[0]['lmList'][0]
        except IndexError:
            #print(f"Error: {e}")
            palm_pos = None

        if palm_pos is not None:
            self.apply_camera_matrix(palm_pos[0], palm_pos[1])
            cv2.circle(img1, (palm_pos[0], palm_pos[1]), 10, (0, 255, 0), thickness=cv2.FILLED)

        cv2.imshow("Image", img1)
        cv2.waitKey(500)

        return palm_pos

    def apply_camera_matrix(self, img_x, img_y):

        # Your palm coordinates in image space
        image_points = np.array([[img_x, img_y]], dtype=np.float32)

        # Camera intrinsic parameters (use your own values)
        fx, fy, cx, cy = 1078, 1077, 317, 306
        camera_matrix = np.array([[fx, 0, cx],
                                  [0, fy, cy],
                                  [0, 0, 1]])

        # Distortion coefficients (use your own values)
        dist_coeffs = np.array([-3.29395596e-01, 6.57199839e+00, 1.06472185e-02, -1.76487035e-02, -3.24388767e+01])

        # Define the transformation matrix
        transformation_matrix = np.linalg.inv(camera_matrix)

        # Add 1.0 as the third coordinate
        image_points = np.insert(image_points, 2, 1.0, axis=1)

        # Apply the transformation matrix
        world_coordinates = np.dot(transformation_matrix, image_points.T).T

        print("World Coordinates:", world_coordinates[:, :2])


class HandPosPublisher(Node):

    HandFinderObj = HandLocationFinder()

    def __init__(self):
        super().__init__('palm_location_finder')

        self.publisher_ = self.create_publisher(String, 'palm_pos_id1', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        control_string = self.HandFinderObj.find_palm_pos()
        #control_string = self.HandRecognition.poll_keys()
        #self.get_logger().info('Ran the Hand Recognition')

        msg = String()
        msg.data = str(control_string)
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)


def main(args=None):
    rclpy.init(args=args)

    palm_pos_publisher = HandPosPublisher()

    rclpy.spin(palm_pos_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    palm_pos_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
