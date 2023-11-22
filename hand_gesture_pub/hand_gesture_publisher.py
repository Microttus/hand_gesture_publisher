import rclpy
from rclpy.node import Node

from std_msgs.msg import String

import os

#from __future__ import print_function

#import roslib; #roslib.load_manifest('champ_teleop')
#import rospy

#from sensor_msgs.msg import Joy
#from geometry_msgs.msg import Twist
#from champ_msgs.msg import Pose as PoseLite
#from geometry_msgs.msg import Pose as Pose
import tensorflow as tf

#import sys, select, termios, tty
import numpy as np

import cv2
import mediapipe as mp
import time
from keras.models import load_model
from cvzone.HandTrackingModule import HandDetector

from ament_index_python.packages import get_package_share_directory

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


#arduinoData = serial.Serial('com7', 9600)


class HandTeleop:
    def test(self):
        #print('Test function')

        success, img1 = cap.read()
        hand, img1 = detector.findHands(img1)
        
        cv2.imshow("Image", img1)
        cv2.waitKey(1)


    def __init__(self):
        package_share_direcroty = get_package_share_directory('hand_gesture_pub');
        print(package_share_direcroty)
        self.model = tf.keras.models.load_model(os.path.join(package_share_direcroty, '/home/martin/haptic_ws/src/hand_gesture_publisher/resource/CNN_LSTM_Latest.h5'))
        

    def poll_keys(self):
        #self.settings = termios.tcgetattr(sys.stdin)

        x = 0
        y = 0
        z = 0
        th = 0
        roll = 0
        pitch = 0
        yaw = 0
        status = 0
        cmd_attempts = 0

        control_msg = 'NaN'

        Data = np.zeros((1, 63, 1))
        pred = np.zeros((3, 1))

        try:
            #print(self.msg)
            #print(self.vels( self.speed, self.turn))
            #print('camera_loop')
            
            #while not rospy.is_shutdown():
                
            success, img = cap.read()
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(imgRGB)

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                        mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

                        for id, lm in enumerate(handLms.landmark):
                            Data[:, id*3] = lm.x
                            Data[:, id*3+1] = lm.y
                            Data[:, id*3+2] = lm.z


            pred = self.model.predict(Data)

            control_msg = 'pred done'

            print(pred[:, 0])

            if pred[:, 0] >=0.7:
                x = 1
                y = 0
                z = 0
                th = 0
                #arduinoData.write(b'1')
                #cv2.putText(img, str("Start"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                control_msg = 'Open'
            if pred[:, 1] >=0.7:
                x = 0
                y = 0
                z = 0
                th = 0
                # arduinoData.write(b'0')
                #cv2.putText(img, str("Stop"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                control_msg = 'Closed'
            '''
            if pred[:, 2] >=0.7:
                x = -1
                y = 0
                z = 0
                th = 0
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                control_msg = 'noe annaet'
            if pred[:, 3] >=0.7:
                x = 0
                y = 0
                z = 0
                th =-1
                control_msg = 'enda noe'
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 4] >= 0.7:
                x = 0
                y = 0
                z = 0
                th =1
                control_msg = 'okoko'
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 5] >=0.7:
                x = 1
                y = 0
                z = 0
                th = 1
                control_msg = 'okoko'
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 6] >=0.7:
                x = 1
                y = 0
                z = 0
                th = -1
                control_msg = 'okoko'
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 7] >=0.7:
                x = -1
                y = 0
                z = 0
                th = -1
                control_msg = 'okoko'
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 8] >=0.7:
                x = -1
                y = 0
                z = 0
                th = 1
                control_msg = 'okoko'
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 9] >=0.7:
                print("Increase Velocity")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 10] >=0.7:
                print("Decrease Velocity")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 11] >=0.7:
                print("Repeat Command")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 12] >=0.7:
                print("Buckle")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 13] >=0.7:
                print("Hold")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 14] >=0.7:
                print("Palmer")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 15] >=0.7:
                print("Spherical")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 16] >=0.7:
                print("Tip")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 17] >=0.7:
                print("Clockwise")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 18] >=0.7:
                print("Anti-Clockwise")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            if pred[:, 19] >=0.7:
                print("Spare")
                #cv2.putText(img, str("Right"), (40, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            '''
            

            #self.velocity_publisher.publish(twist)


        except Exception as e:
            print(e)

        return control_msg

        



class HandGesturePublisher(Node):

    HandRecognition = HandTeleop()

    def __init__(self):
        super().__init__('hand_gesture_publisher')

        self.publisher_ = self.create_publisher(String, 'hand_gestures', 10)
        timer_period = 0.5  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.i = 0

    def timer_callback(self):
        self.HandRecognition.test();
        control_string = self.HandRecognition.poll_keys()
        #self.get_logger().info('Ran the Hand Recognition')

        print(control_string)

        msg = String()
        msg.data = control_string
        self.publisher_.publish(msg)
        #self.get_logger().info('Publishing: "%s"' % msg.data)




def main(args=None):
    rclpy.init(args=args)

    hand_gesture_publisher = HandGesturePublisher()

    rclpy.spin(hand_gesture_publisher)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    hand_gesture_publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()