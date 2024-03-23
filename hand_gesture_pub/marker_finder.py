# Martin Økter
# Marker finder for position tracking
# UiA Grimstad 22/3-24

# ROS2 pkg
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Image pkg
import numpy as np
import cv2
import time


class MarkerLocationFinder:

    def __init__(self):
        self.camera_index = 0
        self.cap = cv2.VideoCapture(self.camera_index)
        self.color_low = 0
        self.color_high = 2

    def find_marker(self):
        palm_pos_coordinates = []
        palm_pos = None

        success, image = self.cap.read()
        #image = cv2.flip(image, -1)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        image_HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_pink = np.array([self.color_low, 120, 120])
        upper_pink = np.array([self.color_high, 255, 255])
        mask = cv2.inRange(image_HSV, lower_pink, upper_pink)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        # findContours returns a list of the outlines of the white shapes in the mask (and a heirarchy that we shall ignore)
        # API differences:
        #   OpenCV 2.x: findContours -> contours, hierarchy
        #   OpenCV 3.x: findContours -> image, contours, hierarchy
        #   OpenCV 4.x: findContours -> contours, hierarchy
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

        # If we have at least one contour, look through each one and pick the biggest
        if len(contours) > 0:
            largest = 0
            area = 0
            for i in range(len(contours)):
                # get the area of the ith contour
                temp_area = cv2.contourArea(contours[i])
                # if it is the biggest we have seen, keep it
                if temp_area > area:
                    area = temp_area
                    largest = i
            # Compute the coordinates of the center of the largest contour
            coordinates = cv2.moments(contours[largest])
            target_x = int(coordinates['m10']/coordinates['m00'])
            target_y = int(coordinates['m01']/coordinates['m00'])

            palm_pos = [target_x, target_y]

            # Pick a suitable diameter for our target (grows with the contour)
            diam = int(np.sqrt(area)/4)
            # draw on a target
            cv2.circle(image, (target_x, target_y), diam, (0, 0, 255), 1)
        cv2.imshow('View', image)
        # Esc key to stop, otherwise repeat after 3 milliseconds
        cv2.waitKey(1)

        if palm_pos is not None:
            palm_pos_coordinates = self.apply_camera_matrix(palm_pos[0], palm_pos[1])

        if palm_pos is not None:
            return palm_pos_coordinates
        else:
            return None

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

        #print("World Coordinates:", world_coordinates[:, :2])
        #print("World Coordinates:", world_coordinates[0, 1])

        return world_coordinates[0, :]


class HandPosPublisher(Node):

    def __init__(self):
        super().__init__('palm_location_finder')

        self.MarkerFinder = MarkerLocationFinder()
        self.detection_flag_ = True

        self.publisher_ = self.create_publisher(Twist, 'marker_pos_id1', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0
        self.gain = 1

    def timer_callback(self):
        palm_coordinates = self.MarkerFinder.find_marker()
        #self.get_logger().info('Ran the Hand Recognition')
        #print(palm_coordinates)

        if palm_coordinates is not None:
            msg = Twist()
            msg.linear.x = palm_coordinates[0] * self.gain
            msg.linear.y = palm_coordinates[1] * self.gain
            self.publisher_.publish(msg)

            if not self.detection_flag_:
                self.get_logger().info("Found marker and started tracking")
                self.detection_flag_ = True
        else:
            if self.detection_flag_:
                self.get_logger().warning("No marker detected ...")
                self.detection_flag_ = False


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